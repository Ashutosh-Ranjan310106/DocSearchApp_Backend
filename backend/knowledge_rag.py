"""
DocSearch Backend — FastAPI server

HUGGINGFACE MIGRATION (on top of all prior fixes)
──────────────────────────────────────────────────
Both AI model integrations are now routed through the HuggingFace Inference API.
Each has its OWN dedicated API key and model env var so they can be managed,
billed, and rotated completely independently.

.env variables added
────────────────────

  # ── Embedding model (replaces Ollama embed) ───────────────────────────────
  HF_EMBED_API_KEY=hf_...          # HuggingFace token with read access
  HF_EMBED_MODEL=sentence-transformers/all-mpnet-base-v2
  #   Any Feature-Extraction model on the HF Hub works here.
  #   The model must return a list[list[float]] (one vector per input text).
  #   Popular choices:
  #     sentence-transformers/all-mpnet-base-v2       → dim 768
  #     sentence-transformers/all-MiniLM-L6-v2        → dim 384
  #     BAAI/bge-large-en-v1.5                        → dim 1024
  #   Set EMBED_DIM to match the chosen model's output dimension.

  # ── LLM / chat model (replaces Ollama chat) ───────────────────────────────
  HF_LLM_API_KEY=hf_...            # separate HF token (can be the same account
                                   # but a different key for independent tracking)
  HF_LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
  #   Any Text-Generation model on the HF Inference API works here.
  #   The model must support the /v1/chat/completions endpoint
  #   (HF's OpenAI-compatible router).  Popular choices:
  #     mistralai/Mistral-7B-Instruct-v0.3
  #     mistralai/Mixtral-8x7B-Instruct-v0.1
  #     meta-llama/Meta-Llama-3.1-8B-Instruct
  #     HuggingFaceH4/zephyr-7b-beta
  HF_LLM_MAX_TOKENS=1024           # max new tokens per LLM response (default 1024)
  HF_LLM_TEMPERATURE=0.7           # sampling temperature (default 0.7)

All other .env variables (EMBED_DIM, EMBED_MAX_TOKENS, UPLOAD_DIR, DB_PATH, …)
remain unchanged and still work exactly as before.

REMOVED env vars (no longer needed — Ollama is gone for inference)
──────────────────────────────────────────────────────────────────
  OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_THINK, EMBED_MODEL

Prior fix list (unchanged — see original comments)
───────────────────────────────────────────────────
FIX #1–#16 are all still applied.  The only changes in this revision are the
two AI back-ends; everything else (routing, persistence, BM25, graph, etc.)
is identical to the previous version.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import mimetypes
import os
import re
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiohttp
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from backend.dockling_document_extraction import chunk_elements, extract_with_dockling
from backend.entity_extractor import extract_rule_entities
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
from backend.lightrag_support import convert_edges, convert_nodes
from backend.hierarchy_kg import inject_hierarchy_edges


load_dotenv()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"WARNING: {name}={raw!r} is not a valid int, using default {default}")
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"WARNING: {name}={raw!r} is not a valid float, using default {default}")
        return default


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── HuggingFace — Embedding model ─────────────────────────────────────────────
# Separate API key so embedding usage is tracked / billed / rotatable
# independently from the LLM.
#
# HF_EMBED_API_KEY  — HuggingFace token with at least read permissions.
#                     Create one at https://huggingface.co/settings/tokens
# HF_EMBED_MODEL    — any Feature-Extraction model on the HF Hub.
#                     Must return list[list[float]] (one vector per input).
# EMBED_DIM         — must match the chosen model's output dimension exactly.
# EMBED_MAX_TOKENS  — max token budget forwarded to LightRAG's EmbeddingFunc.

HF_EMBED_API_KEY  = os.getenv("HF_EMBED_API_KEY", "")
HF_EMBED_MODEL    = os.getenv(
    "HF_EMBED_MODEL",
    "sentence-transformers/all-mpnet-base-v2",   # dim=768 by default
)
EMBED_DIM         = _env_int("EMBED_DIM",        768)
EMBED_MAX_TOKENS  = _env_int("EMBED_MAX_TOKENS", 8192)
# ── Batching & rate-limit config ──────────────────────────────────────────────
# EMBED_BATCH_SIZE  — how many texts to send per API call (default 32).
#                     Smaller = fewer tokens at risk per call (lower cost on failure);
#                     Larger = fewer round-trips (faster for big ingestions).
#                     Set to 0 or negative to disable batching (single call).
# EMBED_BATCH_DELAY — seconds to sleep between consecutive batch calls (default 2.0).
#                     Keeps you under HF Inference API rate limits.
EMBED_BATCH_SIZE  = _env_int(  "EMBED_BATCH_SIZE",   32)
EMBED_BATCH_DELAY = _env_float("EMBED_BATCH_DELAY",   2.0)

# HuggingFace Inference API base URL for Feature-Extraction (embedding) models.
# Format: POST /models/<model_id>  with {"inputs": [...]}
_HF_EMBED_URL = (
    f"https://router.huggingface.co/hf-inference/models/"
    f"{HF_EMBED_MODEL}/pipeline/feature-extraction"
)

# ── HuggingFace — LLM / chat model ────────────────────────────────────────────
# Separate API key from the embedding key — different token, different quota,
# independent rotation.
#
# HF_LLM_API_KEY     — HuggingFace token (read permissions, may differ from
#                      HF_EMBED_API_KEY even if on the same HF account).
# HF_LLM_MODEL       — any Text-Generation model that supports HF's
#                      OpenAI-compatible /v1/chat/completions router.
# HF_LLM_MAX_TOKENS  — maximum new tokens per response (default 1024).
# HF_LLM_TEMPERATURE — sampling temperature (default 0.7).

HF_LLM_API_KEY      = os.getenv("HF_LLM_API_KEY", "")
HF_LLM_MODEL        = os.getenv(
    "HF_LLM_MODEL",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
HF_LLM_MAX_TOKENS   = _env_int("HF_LLM_MAX_TOKENS",      1024)
HF_LLM_TEMPERATURE  = _env_float("HF_LLM_TEMPERATURE",    0.7)

# HuggingFace OpenAI-compatible router endpoint (supports streaming).
# All instruct/chat models on the HF Hub accept requests at this URL.
_HF_LLM_URL = "https://router.huggingface.co/v1/chat/completions"

# ── Other infrastructure config (unchanged) ───────────────────────────────────
UPLOAD_DIR   = Path(os.getenv("UPLOAD_DIR", "./uploads")).resolve()
DB_PATH      = Path(os.getenv("DB_PATH",    "./docsearch.db")).resolve()
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB",   50)
CORS_ORIGINS  = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
]

log               = logging.getLogger("docsearch")
_embed_call_count = 0


# ── Startup validation ────────────────────────────────────────────────────────
# Warn loudly at import time if the API keys are missing so operators
# know immediately what is wrong rather than getting cryptic 401 errors later.

def _warn_missing_key(key_name: str, key_value: str, purpose: str) -> None:
    if not key_value:
        print(
            f"WARNING: {key_name} is not set in .env — "
            f"{purpose} will fail with HTTP 401.  "
            f"Get a token at https://huggingface.co/settings/tokens"
        )


_warn_missing_key("HF_EMBED_API_KEY", HF_EMBED_API_KEY, "embedding")
_warn_missing_key("HF_LLM_API_KEY",   HF_LLM_API_KEY,   "LLM chat")


# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    import spacy
    nlp = spacy.load("en_core_web_md")
    SPACY_OK = True
except Exception:
    SPACY_OK = False

# SentenceTransformer loaded for re-ranking (CrossEncoder) ONLY.
# It is NOT used for document embeddings — all embeddings go through
# the HuggingFace Inference API (_hf_embed) so EMBED_MODEL / EMBED_DIM
# in .env remain the single source of truth.
try:
    from sentence_transformers import CrossEncoder
    _RERANK_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    SENTENCE_TRANSFORMERS_OK = True
except Exception:
    SENTENCE_TRANSFORMERS_OK = False

try:
    import pdfplumber
    PDF_OK = True
except Exception:
    PDF_OK = False

try:
    from docx import Document as DocxDoc
    DOCX_OK = True
except Exception:
    DOCX_OK = False

# LLM availability flag — True when HF_LLM_API_KEY is present.
# We do a lightweight check here; any real 401 / 503 will surface at request time.
LLM_OK = bool(HF_LLM_API_KEY)

# ── In-memory store ───────────────────────────────────────────────────────────
DOCS:   Dict[str, Dict] = {}
CHUNKS: Dict[str, Dict] = {}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING — HuggingFace Feature-Extraction API
# ══════════════════════════════════════════════════════════════════════════════
#
# Single implementation for ALL embedding calls:
#   • embed()               → document ingestion + query embedding at search time
#   • _make_embedding_func() → LightRAG internal embeddings
#
# Credentials used: HF_EMBED_API_KEY + HF_EMBED_MODEL  (env vars)
# The LLM API key (HF_LLM_API_KEY) is NEVER used here.

async def _hf_embed(texts: List[str]) -> np.ndarray:
    """
    Call the HuggingFace Feature-Extraction Inference API with optional batching.

    When EMBED_BATCH_SIZE > 0 the input list is split into chunks of that size
    and each chunk is sent as a separate HTTP request.  EMBED_BATCH_DELAY seconds
    are awaited between consecutive calls so you stay within HF rate limits and
    avoid paying for retries caused by 429 / 503 responses.

    Endpoint:  POST https://router.huggingface.co/hf-inference/models/<model>
    Auth:      Bearer HF_EMBED_API_KEY
    Body:      {"inputs": ["text1", "text2", ...]}
    Response:  list[list[float]]  — one vector per input string

    Raises ValueError if the returned dimension doesn't match EMBED_DIM so
    operators catch model/config mismatches immediately rather than getting
    silent cosine-similarity crashes later.
    """
    global _embed_call_count

    if not HF_EMBED_API_KEY:
        raise RuntimeError(
            "HF_EMBED_API_KEY is not set — cannot call HuggingFace embedding API."
        )

    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)

    # ── Split into batches ────────────────────────────────────────────────────
    batch_size = EMBED_BATCH_SIZE if EMBED_BATCH_SIZE > 0 else len(texts)
    batches: List[List[str]] = [
        texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
    ]

    headers = {
        "Authorization": f"Bearer {HF_EMBED_API_KEY}",
        "Content-Type":  "application/json",
    }

    all_vecs: List[np.ndarray] = []

    async with aiohttp.ClientSession() as session:
        for batch_idx, batch in enumerate(batches):
            _embed_call_count += 1
            call_id = _embed_call_count

            # Polite delay between batches (skip before the very first call)
            if batch_idx > 0 and EMBED_BATCH_DELAY > 0:
                log.debug(
                    "[EMBED] sleeping %.1fs before batch %d/%d",
                    EMBED_BATCH_DELAY, batch_idx + 1, len(batches),
                )
                await asyncio.sleep(EMBED_BATCH_DELAY)

            payload = {
                "inputs":  batch,
                # wait_for_model=True avoids 503 "model loading" errors on cold
                # starts — the request blocks server-side until the model is warm.
                "options": {"wait_for_model": True},
            }

            t0 = time.perf_counter()

            async with session.post(
                _HF_EMBED_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status == 401:
                    raise RuntimeError(
                        "HuggingFace embedding API returned 401 Unauthorized. "
                        "Check HF_EMBED_API_KEY in .env."
                    )
                if resp.status == 429:
                    body = await resp.text()
                    raise RuntimeError(
                        f"HuggingFace embedding API rate-limited (429). "
                        f"Increase EMBED_BATCH_DELAY in .env. Body: {body[:200]}"
                    )
                if resp.status == 503:
                    body = await resp.text()
                    raise RuntimeError(
                        f"HuggingFace embedding model is loading (503). "
                        f"Retry in a few seconds. Body: {body[:200]}"
                    )
                resp.raise_for_status()
                data = await resp.json(content_type=None)

            elapsed = time.perf_counter() - t0

            if not data:
                raise ValueError(
                    f"HuggingFace embedding API returned an empty response "
                    f"for batch {batch_idx + 1}/{len(batches)}."
                )

            # HF Feature-Extraction returns one of:
            #   list[list[float]]          — standard (one vector per input)
            #   list[list[list[float]]]    — token-level; take mean across tokens
            if isinstance(data[0][0], list):
                # shape: (batch, seq_len, dim) → mean over seq_len → (batch, dim)
                batch_arr = np.array(
                    [np.mean(token_vecs, axis=0) for token_vecs in data],
                    dtype=np.float32,
                )
            else:
                batch_arr = np.array(data, dtype=np.float32)

            if batch_arr.ndim == 1:
                batch_arr = batch_arr.reshape(1, -1)

            if batch_arr.shape[1] != EMBED_DIM:
                raise ValueError(
                    f"HF model '{HF_EMBED_MODEL}' returned dim={batch_arr.shape[1]} "
                    f"but EMBED_DIM={EMBED_DIM}. "
                    f"Set EMBED_DIM={batch_arr.shape[1]} in .env and restart."
                )

            log.debug(
                "[EMBED #%d] model=%s batch=%d/%d texts=%d shape=%s %.2fs",
                call_id, HF_EMBED_MODEL,
                batch_idx + 1, len(batches), len(batch),
                batch_arr.shape, elapsed,
            )

            all_vecs.append(batch_arr)

    return np.vstack(all_vecs) if len(all_vecs) > 1 else all_vecs[0]


def _make_embedding_func() -> EmbeddingFunc:
    """EmbeddingFunc for LightRAG — delegates to _hf_embed (HF Inference API)."""
    return EmbeddingFunc(
        embedding_dim=EMBED_DIM,
        max_token_size=EMBED_MAX_TOKENS,
        func=_hf_embed,
    )


async def embed(texts: List[str]) -> np.ndarray:
    """Public async embedding entry-point used by ingestion and search routes."""
    return await _hf_embed(texts)


# ══════════════════════════════════════════════════════════════════════════════
# LLM — HuggingFace OpenAI-compatible chat/completions API
# ══════════════════════════════════════════════════════════════════════════════
#
# Credentials used: HF_LLM_API_KEY + HF_LLM_MODEL  (env vars)
# The embedding API key (HF_EMBED_API_KEY) is NEVER used here.
#
# The HF router at /v1/chat/completions is OpenAI-compatible and supports
# streaming (Server-Sent Events with delta chunks).

async def _hf_chat_complete(
    messages: List[Dict],
    stream: bool = False,
) -> aiohttp.ClientResponse:
    """
    Fire a request to HF's OpenAI-compatible completions endpoint.

    Returns the raw aiohttp response object so callers can consume
    the body incrementally (streaming) or all at once (non-streaming).
    The caller is responsible for closing the response / session.
    """
    if not HF_LLM_API_KEY:
        raise RuntimeError(
            "HF_LLM_API_KEY is not set — cannot call HuggingFace LLM API."
        )

    headers = {
        "Authorization": f"Bearer {HF_LLM_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload: Dict[str, Any] = {
        "model":       HF_LLM_MODEL,
        "messages":    messages,
        "max_tokens":  HF_LLM_MAX_TOKENS,
        "temperature": HF_LLM_TEMPERATURE,
        "stream":      stream,
    }

    # We return the response without closing it so the caller can stream.
    # Use a long timeout: large models can take 60–120 s to generate.
    session = aiohttp.ClientSession()
    resp    = await session.post(
        _HF_LLM_URL,
        headers=headers,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=300),
    )

    if resp.status == 401:
        await session.close()
        raise RuntimeError(
            "HuggingFace LLM API returned 401 Unauthorized. "
            "Check HF_LLM_API_KEY in .env."
        )
    if resp.status not in (200, 201):
        body = await resp.text()
        await session.close()
        raise RuntimeError(
            f"HuggingFace LLM API error {resp.status}: {body[:400]}"
        )

    # Attach session to response so callers can close both together.
    resp._hf_session = session   # type: ignore[attr-defined]
    return resp


async def _hf_chat_blocking(messages: List[Dict]) -> str:
    """
    Non-streaming LLM call.  Returns the full assistant response as a string.
    Used by lightrag_hf() and the non-streaming /chat endpoint.
    """
    resp = await _hf_chat_complete(messages, stream=False)
    try:
        data = await resp.json(content_type=None)
    finally:
        await resp._hf_session.close()   # type: ignore[attr-defined]

    # Standard OpenAI-compatible response shape.
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected HF LLM response shape: {json.dumps(data)[:300]}"
        ) from exc


async def _hf_chat_stream(messages: List[Dict]) -> asyncio.Queue:
    """
    Streaming LLM call.  Yields chunks into an asyncio.Queue in the same
    format as the old stream_llm() so all downstream code is unchanged:

        {"type": "text",  "text": "..."}   — token chunk
        {"type": "error", "text": "..."}   — error string
        None                               — end sentinel

    SSE lines from HF look like:
        data: {"choices":[{"delta":{"content":"tok"}}]}
        data: [DONE]
    """
    q: asyncio.Queue = asyncio.Queue()

    async def _run() -> None:
        try:
            resp = await _hf_chat_complete(messages, stream=True)
            session = resp._hf_session  # type: ignore[attr-defined]
            try:
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload_str = line[len("data:"):].strip()
                    if payload_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload_str)
                        delta = chunk["choices"][0]["delta"]
                        text  = delta.get("content", "")
                        if text:
                            await q.put({"type": "text", "text": text})
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Malformed chunk — skip silently; model may emit
                        # keep-alive comments or partial lines.
                        continue
            finally:
                await session.close()
        except Exception as exc:
            await q.put({"type": "error", "text": str(exc)})

        await q.put(None)  # end sentinel

    asyncio.create_task(_run())
    return q


# ── LightRAG LLM adapter ──────────────────────────────────────────────────────
# LightRAG calls this with (prompt, system_prompt, history_messages, **kwargs).
# We forward it to the HF LLM using HF_LLM_API_KEY (not the embed key).

async def lightrag_hf(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    keyword_extraction: bool = False,   # injected by LightRAG — accepted, ignored
    **kwargs,
) -> str:
    messages: List[Dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    messages.append({"role": "user", "content": prompt})
    return await _hf_chat_blocking(messages)


# ── stream_llm (public entry-point for /chat/stream and direct LLM calls) ─────
# Signature is identical to the old Ollama version so all callers are unchanged.

async def stream_llm(system: str, messages: List[Dict]) -> asyncio.Queue:
    """
    Start a streaming HF LLM request.

    Returns an asyncio.Queue that yields:
        {"type": "text",  "text": "..."}
        {"type": "error", "text": "..."}
        None   (end sentinel)
    """
    if not LLM_OK:
        q: asyncio.Queue = asyncio.Queue()
        await q.put({"type": "text", "text": "[LLM not configured — set HF_LLM_API_KEY]"})
        await q.put(None)
        return q

    hf_messages = [{"role": "system", "content": system}, *messages]
    return await _hf_chat_stream(hf_messages)


# ══════════════════════════════════════════════════════════════════════════════
# LightRAG initialisation  (FIX #1 / #2 / #3 / #4 / #8)
# ══════════════════════════════════════════════════════════════════════════════

_rag: LightRAG | None = None


async def _init_rag() -> LightRAG:
    global _rag
    if _rag is not None:
        return _rag
    _rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=lightrag_hf,          # HF LLM, HF_LLM_API_KEY
        embedding_func=_make_embedding_func(),  # HF embed, HF_EMBED_API_KEY
    )
    await _rag.initialize_storages()
    await initialize_pipeline_status()
    return _rag


async def _get_rag() -> LightRAG:
    if _rag is None:
        raise RuntimeError("LightRAG not yet initialised — startup did not complete")
    return _rag


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    _load_all_from_db()
    await _init_rag()
    yield
    if _rag is not None:
        await _rag.finalize_storages()


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocSearch API",
    description="Hybrid RAG system — HuggingFace Inference API for embeddings and LLM",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── File storage helpers ──────────────────────────────────────────────────────
def _safe_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w\-. ()]", "_", name).strip()
    return name or "upload"


def _save_uploaded_file(filename: str, content: bytes) -> Path:
    safe_name = _safe_filename(filename)
    stem, ext = os.path.splitext(safe_name)
    dest      = UPLOAD_DIR / safe_name
    counter   = 1
    while dest.exists():
        dest = UPLOAD_DIR / f"{stem} ({counter}){ext}"
        counter += 1
    dest.write_bytes(content)
    return dest


# ── Persistence (SQLite) ──────────────────────────────────────────────────────
def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_db() -> None:
    conn = _get_db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS docs (
                doc_id          TEXT PRIMARY KEY,
                filename        TEXT NOT NULL,
                size_bytes      INTEGER NOT NULL,
                char_count      INTEGER NOT NULL,
                chunk_count     INTEGER NOT NULL,
                entities_json   TEXT NOT NULL,
                uploaded_at     TEXT NOT NULL,
                text_preview    TEXT,
                file_path       TEXT NOT NULL,
                file_url        TEXT NOT NULL,
                stored_filename TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id            TEXT PRIMARY KEY,
                doc_id              TEXT NOT NULL REFERENCES docs(doc_id) ON DELETE CASCADE,
                doc_name            TEXT,
                text                TEXT NOT NULL,
                display_text        TEXT,
                chunk_index         INTEGER,
                breadcrumb          TEXT,
                hierarchy_path_json TEXT,
                table_part          INTEGER,
                table_parts_total   INTEGER,
                prev_chunk_id       TEXT,
                next_chunk_id       TEXT,
                page_hint           TEXT,
                file_url            TEXT,
                embedding_blob      BLOB,
                embedding_dim       INTEGER,
                entities_json       TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
        conn.commit()
    finally:
        conn.close()


def _sql_safe(value):
    if value is None or isinstance(value, (int, float, str, bytes)):
        return value
    return json.dumps(value)


def _persist_document(doc: Dict, chunk_objs: List[Dict]) -> None:
    conn = _get_db()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO docs
                (doc_id, filename, size_bytes, char_count, chunk_count,
                 entities_json, uploaded_at, text_preview, file_path,
                 file_url, stored_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc["doc_id"], doc["filename"], doc["size_bytes"],
                doc["char_count"], doc["chunk_count"],
                json.dumps(doc["entities"]), doc["uploaded_at"],
                _sql_safe(doc.get("text_preview")), doc["file_path"],
                doc["file_url"], doc["stored_filename"],
            ),
        )
        for c in chunk_objs:
            vec     = c.get("embedding")
            emb_arr = np.asarray(vec, dtype=np.float32) if vec is not None else None
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks
                    (chunk_id, doc_id, doc_name, text, display_text,
                     chunk_index, breadcrumb, hierarchy_path_json,
                     table_part, table_parts_total, prev_chunk_id,
                     next_chunk_id, page_hint, file_url, embedding_blob,
                     embedding_dim, entities_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    c["chunk_id"], c["doc_id"], _sql_safe(c.get("doc_name")),
                    c["text"], _sql_safe(c.get("display_text")),
                    c.get("index"), _sql_safe(c.get("breadcrumb")),
                    json.dumps(c.get("hierarchy_path")), c.get("table_part"),
                    c.get("table_parts_total"), _sql_safe(c.get("prev_chunk_id")),
                    _sql_safe(c.get("next_chunk_id")), _sql_safe(c.get("page_hint")),
                    _sql_safe(c.get("file_url")),
                    emb_arr.tobytes() if emb_arr is not None else None,
                    emb_arr.shape[0] if emb_arr is not None else None,
                    json.dumps(c.get("entities", [])),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _delete_document_row(doc_id: str) -> None:
    conn = _get_db()
    try:
        conn.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))
        conn.commit()
    finally:
        conn.close()


def _row_to_doc(row: sqlite3.Row, chunk_ids: List[str]) -> Dict:
    return {
        "doc_id":          row["doc_id"],
        "filename":        row["filename"],
        "size_bytes":      row["size_bytes"],
        "char_count":      row["char_count"],
        "chunk_count":     row["chunk_count"],
        "chunks":          chunk_ids,
        "entities":        json.loads(row["entities_json"]),
        "uploaded_at":     row["uploaded_at"],
        "text_preview":    row["text_preview"],
        "file_path":       row["file_path"],
        "file_url":        row["file_url"],
        "stored_filename": row["stored_filename"],
    }


def _row_to_chunk(row: sqlite3.Row) -> Dict:
    embedding = None
    if row["embedding_blob"] is not None:
        embedding = np.frombuffer(row["embedding_blob"], dtype=np.float32).tolist()
    return {
        "chunk_id":          row["chunk_id"],
        "doc_id":            row["doc_id"],
        "doc_name":          row["doc_name"],
        "text":              row["text"],
        "display_text":      row["display_text"],
        "index":             row["chunk_index"],
        "breadcrumb":        row["breadcrumb"],
        "hierarchy_path":    json.loads(row["hierarchy_path_json"]) if row["hierarchy_path_json"] else None,
        "table_part":        row["table_part"],
        "table_parts_total": row["table_parts_total"],
        "prev_chunk_id":     row["prev_chunk_id"],
        "next_chunk_id":     row["next_chunk_id"],
        "page_hint":         row["page_hint"],
        "file_url":          row["file_url"],
        "embedding":         embedding,
        "entities":          json.loads(row["entities_json"]) if row["entities_json"] else [],
    }


def _load_all_from_db() -> None:
    conn = _get_db()
    try:
        chunk_rows = conn.execute(
            "SELECT * FROM chunks ORDER BY doc_id, chunk_index"
        ).fetchall()
        chunks_by_doc: Dict[str, List[str]] = {}
        for row in chunk_rows:
            chunk = _row_to_chunk(row)
            CHUNKS[chunk["chunk_id"]] = chunk
            chunks_by_doc.setdefault(chunk["doc_id"], []).append(chunk["chunk_id"])

        doc_rows = conn.execute("SELECT * FROM docs").fetchall()
        for row in doc_rows:
            doc_id = row["doc_id"]
            DOCS[doc_id] = _row_to_doc(row, chunks_by_doc.get(doc_id, []))
    finally:
        conn.close()


# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []
    top_k: int = 5
    use_reranking: bool = True
    search_mode: str = "hybrid"
    graph_weight: float = 0.3


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    search_mode: str = "hybrid"
    doc_ids: Optional[List[str]] = None
    graph_weight: float = 0.3


class EntitySearchRequest(BaseModel):
    entities: List[str]
    top_k: int = 10


# ── Cosine similarity (FIX #10) ───────────────────────────────────────────────
def cosine_sim(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or query_vec.ndim != 1:
        raise ValueError(
            f"cosine_sim expects 1-D query and 2-D matrix, "
            f"got {query_vec.shape} and {matrix.shape}"
        )
    return np.dot(matrix, query_vec)


# ── BM25 ──────────────────────────────────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def bm25_score(
    query_tokens: List[str],
    corpus_chunks: List[Dict],
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    N = len(corpus_chunks)
    if N == 0:
        return np.array([])
    doc_lens = [len(tokenize(c["text"])) for c in corpus_chunks]
    avgdl    = sum(doc_lens) / N if N else 1

    df: Dict[str, int] = {}
    tf_per_doc: List[Dict[str, int]] = []
    for c in corpus_chunks:
        toks: Dict[str, int] = {}
        for t in tokenize(c["text"]):
            toks[t] = toks.get(t, 0) + 1
        tf_per_doc.append(toks)
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    scores = np.zeros(N)
    for qt in query_tokens:
        idf = math.log((N - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5) + 1)
        for i, (tf, dl) in enumerate(zip(tf_per_doc, doc_lens)):
            f = tf.get(qt, 0)
            scores[i] += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
    return scores


# ── Reranking ─────────────────────────────────────────────────────────────────
def rerank(query: str, candidates: List[Dict]) -> List[Dict]:
    if not SENTENCE_TRANSFORMERS_OK or not candidates:
        return candidates
    pairs  = [(query, c["text"]) for c in candidates]
    scores = _RERANK_MODEL.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)


# ── Graph search (LightRAG) ───────────────────────────────────────────────────
async def _graph_search(
    query: str,
    top_k: int = 10,
    doc_ids: Optional[List[str]] = None,
) -> List[Dict]:
    rag = _rag
    if rag is None:
        return []
    try:
        answer_text: str = await rag.aquery(
            query, param=QueryParam(mode="mix", top_k=min(top_k * 4, 60))
        )
    except Exception as exc:
        log.warning("LightRAG graph query failed (non-fatal): %s", exc)
        return []

    if not answer_text or answer_text.strip().lower().startswith("sorry"):
        return []

    answer_toks = set(tokenize(answer_text))
    if not answer_toks:
        return []

    pool = [
        c for c in CHUNKS.values()
        if (doc_ids is None or c["doc_id"] in doc_ids)
        and c.get("embedding") is not None
    ]
    if not pool:
        return []

    raw_scores: List[float] = []
    for c in pool:
        chunk_toks = set(tokenize(c["text"]))
        overlap    = len(chunk_toks & answer_toks)
        raw_scores.append(overlap / len(answer_toks))

    arr = np.array(raw_scores, dtype=np.float32)
    mx  = arr.max()
    if mx > 0:
        arr /= mx

    top_idx    = np.argsort(arr)[::-1][: top_k]
    candidates = []
    for i in top_idx:
        if arr[i] == 0:
            break
        c                = dict(pool[i])
        c["graph_score"] = float(arr[i])
        c["score"]       = float(arr[i])
        c["sem_score"]   = 0.0
        c["bm25_score"]  = 0.0
        candidates.append(c)

    return candidates


# ── Hybrid search ─────────────────────────────────────────────────────────────
async def hybrid_search(
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    doc_ids: Optional[List[str]] = None,
    graph_weight: float = 0.3,
) -> List[Dict]:
    pool = [
        c for c in CHUNKS.values()
        if doc_ids is None or c["doc_id"] in doc_ids
    ]
    if not pool:
        return []

    pool = [c for c in pool if c.get("embedding") is not None]
    if not pool:
        return []

    if mode == "graph":
        return []

    query_vec  = (await embed([query]))[0]
    query_toks = tokenize(query)

    sem_scores  = np.zeros(len(pool))
    bm25_scores = np.zeros(len(pool))

    if mode in ("hybrid", "semantic", "full"):
        vecs       = np.array([c["embedding"] for c in pool], dtype=np.float32)
        sem_scores = cosine_sim(query_vec, vecs)

    if mode in ("hybrid", "keyword", "full"):
        bm25_scores = bm25_score(query_toks, pool)
        mx = bm25_scores.max()
        if mx > 0:
            bm25_scores /= mx

    alpha    = 1.0 if mode == "semantic" else 0.0 if mode == "keyword" else 0.5
    combined = alpha * sem_scores + (1 - alpha) * bm25_scores

    top_idx    = np.argsort(combined)[::-1][: top_k * 2]
    candidates = []
    for i in top_idx:
        c                = dict(pool[i])
        c["score"]       = float(combined[i])
        c["sem_score"]   = float(sem_scores[i])
        c["bm25_score"]  = float(bm25_scores[i])
        c["graph_score"] = 0.0
        candidates.append(c)

    return candidates[:top_k]


async def hybrid_search_async(
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    doc_ids: Optional[List[str]] = None,
    graph_weight: float = 0.3,
) -> List[Dict]:
    if mode == "graph":
        results = await _graph_search(query, top_k=top_k * 2, doc_ids=doc_ids)
        return results[:top_k]

    if mode != "full":
        return await hybrid_search(query, top_k=top_k, mode=mode, doc_ids=doc_ids)

    gw        = max(0.0, min(1.0, graph_weight))
    vk_weight = 1.0 - gw

    vec_results, graph_results = await asyncio.gather(
        hybrid_search(query, top_k * 2, "hybrid", doc_ids),
        _graph_search(query, top_k=top_k * 2, doc_ids=doc_ids),
    )

    scores: Dict[str, Dict] = {}
    for c in vec_results:
        cid           = c["chunk_id"]
        scores[cid]   = dict(c)
        scores[cid]["combined_score"] = vk_weight * c["score"]

    for c in graph_results:
        cid = c["chunk_id"]
        gs  = gw * c["graph_score"]
        if cid in scores:
            scores[cid]["graph_score"]    = c["graph_score"]
            scores[cid]["combined_score"] = scores[cid].get("combined_score", 0.0) + gs
        else:
            entry                   = dict(c)
            entry["combined_score"] = gs
            scores[cid]             = entry

    merged = sorted(scores.values(), key=lambda x: x["combined_score"], reverse=True)
    for m in merged:
        m["score"] = m.pop("combined_score")

    return merged[:top_k]


# ── Citation builder ──────────────────────────────────────────────────────────
def build_citations(chunks: List[Dict]) -> List[Dict]:
    citations = []
    for i, c in enumerate(chunks):
        display = c.get("display_text") or c["text"]
        citations.append({
            "citation_id": f"[{i + 1}]",
            "chunk_id":    c["chunk_id"],
            "doc_id":      c["doc_id"],
            "doc_name":    c.get("doc_name", ""),
            "chunk_index": c.get("index", 0),
            "char_start":  c.get("char_start", 0),
            "score":       c.get("rerank_score", c.get("score", 0)),
            "snippet":     display[:300],
            "entities":    c.get("entities", []),
        })
    return citations


# ── Shared context builder ────────────────────────────────────────────────────
async def _build_context(req: ChatRequest):
    candidates = await hybrid_search_async(
        req.query, req.top_k * 2, req.search_mode,
        doc_ids=None, graph_weight=req.graph_weight,
    )
    if req.use_reranking:
        candidates = rerank(req.query, candidates)
    candidates = candidates[: req.top_k]
    citations  = build_citations(candidates)

    context_blocks = []
    for i, c in enumerate(candidates):
        gs         = c.get("graph_score", 0.0)
        source_tag = f"doc: {c.get('doc_name', '?')}"
        if gs > 0:
            source_tag += f", graph_score: {gs:.2f}"
        context_blocks.append(f"[{i + 1}] ({source_tag})\n{c['text']}")
    context = "\n\n".join(context_blocks)

    system = (
        "You are a precise RAG assistant with access to a knowledge graph and document chunks. "
        "Answer the question using ONLY the numbered context blocks below. "
        "Cite every claim inline with [N] notation matching the block number. "
        "Blocks tagged 'graph_score' were retrieved from a knowledge graph and may contain "
        "synthesised relationships — weigh them alongside direct text evidence. "
        "If the context does not contain enough information, say so explicitly."
    )
    messages = [
        *[{"role": m.role, "content": m.content} for m in req.history],
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"},
    ]
    return candidates, citations, context, system, messages


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

# ── Documents ─────────────────────────────────────────────────────────────────

@app.post("/documents/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB} MB)")

    rag = await _get_rag()

    doc_id      = str(uuid.uuid4())
    stored_path = _save_uploaded_file(file.filename, content)
    file_url    = f"/documents/{doc_id}/file"

    elements   = extract_with_dockling(file.filename, content)
    chunks_raw = chunk_elements(elements, target_size=1200, overlap=150)

    text        = "\n\n".join(el.get("text", "") for el in elements if el.get("text"))
    chunk_texts = [c["text"] for c in chunks_raw]
    embeddings  = await _hf_embed(chunk_texts)   # HF embed API, HF_EMBED_API_KEY

    chunk_objs: List[Dict] = []
    doc_kg: Dict = {"chunks": [], "relationships": [], "entities": []}
    all_entities: Dict[str, str] = {}

    for raw, vec in zip(chunks_raw, embeddings):
        cid = str(uuid.uuid4())

        nodes, edges = extract_rule_entities(
            text=raw["text"],
            chunk_key=cid,
            file_path=file.filename,
            timestamp=int(time.time()),
            table_data=raw.get("data"),
        )

        nodes, edges = inject_hierarchy_edges(
            nodes=nodes,
            edges=edges,
            hierarchy_path=raw.get("hierarchy_path"),
            chunk_id=cid,
            file_path=file.filename,
        )

        doc_kg["chunks"].append({
            "content":   raw["text"],
            "source_id": cid,
            "file_path": file.filename,
        })
        doc_kg["entities"].extend(convert_nodes(nodes))
        doc_kg["relationships"].extend(convert_edges(edges))

        ents: List[Dict] = []
        for entity_name, entity_list in nodes.items():
            entity = entity_list[0]
            ents.append({"text": entity_name, "label": entity["entity_type"]})
            all_entities[entity_name] = entity["entity_type"]

        chunk_objs.append({
            "chunk_id":          cid,
            "doc_id":            doc_id,
            "doc_name":          file.filename,
            "text":              raw["text"],
            "display_text":      raw.get("display_text", raw["text"]),
            "index":             raw["index"],
            "breadcrumb":        raw.get("breadcrumb"),
            "hierarchy_path":    raw.get("hierarchy_path"),
            "table_part":        raw.get("table_part"),
            "table_parts_total": raw.get("table_parts_total"),
            "prev_chunk_id":     None,
            "next_chunk_id":     None,
            "page_hint":         raw.get("page"),
            "file_url":          file_url,
            "embedding":         vec.tolist(),
            "entities":          ents,
        })

    # FIX #13 — fill prev/next before persisting
    index_to_cid = {c["index"]: c["chunk_id"] for c in chunk_objs}
    for obj in chunk_objs:
        idx = obj["index"]
        obj["prev_chunk_id"] = index_to_cid.get(idx - 1)
        obj["next_chunk_id"] = index_to_cid.get(idx + 1)
        CHUNKS[obj["chunk_id"]] = obj

    DOCS[doc_id] = {
        "doc_id":          doc_id,
        "filename":        file.filename,
        "size_bytes":      len(content),
        "char_count":      len(text),
        "chunk_count":     len(chunk_objs),
        "chunks":          [c["chunk_id"] for c in chunk_objs],
        "entities":        all_entities,
        "uploaded_at":     datetime.utcnow().isoformat(),
        "text_preview":    text[:500],
        "file_path":       str(stored_path),
        "file_url":        file_url,
        "stored_filename": stored_path.name,
    }

    persisted = True
    try:
        _persist_document(DOCS[doc_id], chunk_objs)
    except Exception as e:
        import traceback
        persisted = False
        print(f"ERROR: failed to persist {doc_id}: {e}")
        traceback.print_exc()

    await rag.ainsert_custom_kg(doc_kg)

    return {
        "doc_id":       doc_id,
        "filename":     file.filename,
        "chunk_count":  len(chunk_objs),
        "char_count":   len(text),
        "top_entities": list(all_entities.items())[:20],
        "file_url":     file_url,
        "persisted":    persisted,
    }


@app.get("/documents", tags=["Documents"])
def list_documents():
    return [
        {k: v for k, v in d.items() if k != "text_preview"}
        for d in DOCS.values()
    ]


@app.get("/documents/{doc_id}", tags=["Documents"])
def get_document(doc_id: str):
    if doc_id not in DOCS:
        raise HTTPException(404, "Document not found")
    return DOCS[doc_id]


@app.get("/documents/{doc_id}/file")
def get_document_file(doc_id: str):
    if doc_id not in DOCS:
        raise HTTPException(404, "Document not found")
    doc         = DOCS[doc_id]
    stored_path = Path(doc["file_path"])
    if not stored_path.exists():
        raise HTTPException(404, "Stored file is missing on disk")

    mime, _ = mimetypes.guess_type(str(stored_path))
    mime    = mime or "application/octet-stream"

    return FileResponse(
        path=doc["file_path"],
        media_type=mime,
        headers={"Content-Disposition": f'inline; filename="{doc["filename"]}"'},
    )


@app.get("/documents/{doc_id}/chunks", tags=["Documents"])
def get_document_chunks(doc_id: str, page: int = 0, size: int = 20):
    if doc_id not in DOCS:
        raise HTTPException(404, "Document not found")
    doc       = DOCS[doc_id]
    chunk_ids = doc["chunks"]
    start     = page * size
    page_ids  = chunk_ids[start: start + size]
    chunks    = [
        {k: v for k, v in CHUNKS[cid].items() if k != "embedding"}
        for cid in page_ids
        if cid in CHUNKS
    ]
    return {
        "doc_id":   doc_id,
        "total":    len(chunk_ids),
        "page":     page,
        "size":     size,
        "file_url": doc.get("file_url"),
        "chunks":   chunks,
    }


@app.get("/documents/{doc_id}/chunks/{chunk_id}", tags=["Documents"])
def get_chunk(doc_id: str, chunk_id: str):
    c = CHUNKS.get(chunk_id)
    if not c or c["doc_id"] != doc_id:
        raise HTTPException(404, "Chunk not found")
    return {k: v for k, v in c.items() if k != "embedding"}


@app.delete("/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: str):
    if doc_id not in DOCS:
        raise HTTPException(404, "Document not found")
    doc = DOCS[doc_id]
    for cid in doc["chunks"]:
        CHUNKS.pop(cid, None)

    stored_path  = Path(doc["file_path"])
    file_deleted = False
    if stored_path.exists():
        try:
            stored_path.unlink()
            file_deleted = True
        except OSError:
            pass

    del DOCS[doc_id]
    try:
        _delete_document_row(doc_id)
    except Exception as e:
        print(f"WARNING: failed to delete {doc_id} from SQLite: {e}")

    return {"deleted": doc_id, "file_deleted": file_deleted}


# ── Search ────────────────────────────────────────────────────────────────────

@app.post("/search", tags=["Search"])
async def search(req: SearchRequest):
    candidates = await hybrid_search_async(
        req.query, req.top_k * 2, req.search_mode, req.doc_ids, req.graph_weight
    )
    if not candidates:
        return {"query": req.query, "results": [], "citations": []}
    if SENTENCE_TRANSFORMERS_OK:
        candidates = rerank(req.query, candidates)
    candidates = candidates[: req.top_k]
    citations  = build_citations(candidates)
    results    = [{k: v for k, v in c.items() if k != "embedding"} for c in candidates]
    return {
        "query":     req.query,
        "mode":      req.search_mode,
        "count":     len(results),
        "results":   results,
        "citations": citations,
    }


@app.post("/search/entities", tags=["Search"])
def entity_search(req: EntitySearchRequest):
    entity_lower = [e.lower() for e in req.entities]
    matched      = []
    for c in CHUNKS.values():
        chunk_ents = [e["text"].lower() for e in c.get("entities", [])]
        hits = sum(1 for e in entity_lower if any(e in ce for ce in chunk_ents))
        if hits > 0:
            obj                = {k: v for k, v in c.items() if k != "embedding"}
            obj["entity_hits"] = hits
            matched.append(obj)
    matched.sort(key=lambda x: x["entity_hits"], reverse=True)
    return {
        "entities": req.entities,
        "count":    len(matched[: req.top_k]),
        "results":  matched[: req.top_k],
    }


@app.get("/entities", tags=["Search"])
def list_entities(doc_id: Optional[str] = None):
    agg: Dict[str, Dict] = {}
    for c in CHUNKS.values():
        if doc_id and c["doc_id"] != doc_id:
            continue
        for e in c.get("entities", []):
            key = e["text"]
            if key not in agg:
                agg[key] = {"text": key, "label": e["label"], "count": 0, "doc_ids": set()}
            agg[key]["count"] += 1
            agg[key]["doc_ids"].add(c["doc_id"])
    result = [
        {"text": v["text"], "label": v["label"], "count": v["count"], "doc_ids": list(v["doc_ids"])}
        for v in sorted(agg.values(), key=lambda x: x["count"], reverse=True)
    ]
    return {"total": len(result), "entities": result[:200]}


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat", tags=["Chat"])
async def chat(req: ChatRequest):
    candidates, citations, context, system, messages = await _build_context(req)

    if not LLM_OK:
        return {
            "answer":      f"[LLM not available — set HF_LLM_API_KEY] Retrieved {len(candidates)} chunks.",
            "citations":   citations,
            "chunks_used": len(candidates),
        }

    hf_messages = [{"role": "system", "content": system}, *messages]
    answer      = await _hf_chat_blocking(hf_messages)
    return {
        "answer":      answer,
        "citations":   citations,
        "chunks_used": len(candidates),
    }


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(req: ChatRequest):
    candidates, citations, context, system, messages = await _build_context(req)

    async def event_gen():
        yield f"event: citations\ndata: {json.dumps(citations)}\n\n"

        if not LLM_OK:
            yield (
                "data: "
                + json.dumps({"text": "[LLM not configured — set HF_LLM_API_KEY in .env]"})
                + "\n\n"
            )
            yield "event: done\ndata: {}\n\n"
            return

        hf_messages = [{"role": "system", "content": system}, *messages]
        q           = await _hf_chat_stream(hf_messages)

        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if item["type"] == "text":
                    yield "data: " + json.dumps({"text": item["text"]}) + "\n\n"
                elif item["type"] == "error":
                    yield "event: error\ndata: " + json.dumps({"error": item["text"]}) + "\n\n"
                    return
        except asyncio.CancelledError:
            return

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── Graph export ──────────────────────────────────────────────────────────────
@app.get("/graph/export", tags=["Graph"])
async def graph_export(
    doc_id: Optional[str]  = Query(None),
    max_nodes: int         = Query(2000, ge=1, le=10000),
    max_edges: int         = Query(5000, ge=1, le=50000),
    min_degree: int        = Query(0,    ge=0),
):
    rag = _rag
    if rag is None:
        return {"nodes": [], "edges": [], "meta": {
            "total_nodes": 0, "total_edges": 0,
            "returned_nodes": 0, "returned_edges": 0,
            "truncated": False, "doc_id_filter": doc_id,
        }}

    try:
        raw_nodes, raw_edges = await asyncio.gather(
            rag.chunk_entity_relation_graph.get_all_nodes(),
            rag.chunk_entity_relation_graph.get_all_edges(),
        )
    except Exception as exc:
        log.warning("graph_export: failed to read graph storage: %s", exc)
        return {"nodes": [], "edges": [], "meta": {
            "total_nodes": 0, "total_edges": 0,
            "returned_nodes": 0, "returned_edges": 0,
            "truncated": False, "doc_id_filter": doc_id,
        }}

    degree: Dict[str, int] = {}
    for e in raw_edges:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1

    filter_file_paths: Optional[Set[str]] = None
    if doc_id is not None:
        doc_meta = DOCS.get(doc_id)
        if doc_meta is None:
            for _id, meta in DOCS.items():
                if meta.get("filename") == doc_id:
                    doc_meta = meta
                    break
        if doc_meta is None:
            raise HTTPException(404, f"Document '{doc_id}' not found")

        filename: Optional[str] = (
            doc_meta.get("filename") or doc_meta.get("file_path") or doc_meta.get("name")
        )
        if not filename:
            filename = doc_id
        filter_file_paths = {filename, os.path.basename(filename)}

    def _passes_filter(node_or_edge: Dict) -> bool:
        if filter_file_paths is None:
            return True
        fp: str = node_or_edge.get("file_path", "")
        if not fp:
            return False
        return fp in filter_file_paths or os.path.basename(fp) in filter_file_paths

    chunk_to_doc: Dict[str, str] = {cid: c["doc_id"] for cid, c in CHUNKS.items()}

    def _source_id_str_to_doc_ids(source_id_str: str) -> List[str]:
        parts = [s.strip() for s in source_id_str.split(",") if s.strip()]
        return list({chunk_to_doc[p] for p in parts if p in chunk_to_doc})

    all_candidate_nodes: List[Dict] = []
    for n in raw_nodes:
        nid         = n.get("id", "")
        source_str  = n.get("source_id", n.get("source_ids", ""))
        node_degree = degree.get(nid, 0)
        if node_degree < min_degree:
            continue
        if not _passes_filter(n):
            continue
        all_candidate_nodes.append({
            "id":          nid,
            "label":       n.get("entity_name", nid),
            "type":        n.get("entity_type", "UNKNOWN"),
            "description": n.get("description", ""),
            "degree":      node_degree,
            "file_path":   n.get("file_path", ""),
            "doc_ids":     _source_id_str_to_doc_ids(source_str),
            "source_ids":  [s.strip() for s in source_str.split(",") if s.strip()],
        })

    total_nodes_before_cap = len(all_candidate_nodes)

    if doc_id is None and all_candidate_nodes:
        by_file: Dict[str, List[Dict]] = {}
        for node in all_candidate_nodes:
            key = os.path.basename(node["file_path"]) or "__unknown__"
            by_file.setdefault(key, []).append(node)

        n_docs = len(by_file)
        if n_docs <= 1:
            nodes_out = sorted(all_candidate_nodes, key=lambda x: x["degree"], reverse=True)[:max_nodes]
        else:
            base_quota    = max_nodes // n_docs
            remainder     = max_nodes - base_quota * n_docs
            for key in by_file:
                by_file[key].sort(key=lambda x: x["degree"], reverse=True)

            selected: List[Dict]    = []
            surplus_slots: int      = remainder
            overflow_docs: List[str] = []

            for key, doc_nodes in by_file.items():
                take      = min(base_quota, len(doc_nodes))
                selected += doc_nodes[:take]
                leftover  = len(doc_nodes) - take
                if leftover > 0:
                    overflow_docs.append(key)
                else:
                    surplus_slots += base_quota - take

            if surplus_slots > 0 and overflow_docs:
                extra_per_doc = max(1, surplus_slots // len(overflow_docs))
                for key in overflow_docs:
                    if surplus_slots <= 0:
                        break
                    doc_nodes  = by_file[key]
                    already    = min(base_quota, len(doc_nodes))
                    extra_take = min(extra_per_doc, len(doc_nodes) - already, surplus_slots)
                    selected  += doc_nodes[already: already + extra_take]
                    surplus_slots -= extra_take

            nodes_out = sorted(selected, key=lambda x: x["degree"], reverse=True)
    else:
        nodes_out = sorted(all_candidate_nodes, key=lambda x: x["degree"], reverse=True)[:max_nodes]

    total_nodes = total_nodes_before_cap
    capped_ids  = {n["id"] for n in nodes_out}

    edges_out: List[Dict] = []
    for e in raw_edges:
        src        = e.get("source", "")
        tgt        = e.get("target", "")
        source_str = e.get("source_id", e.get("source_ids", ""))
        if src not in capped_ids or tgt not in capped_ids:
            continue
        if not _passes_filter(e):
            continue
        edges_out.append({
            "id":          f"{src}||{tgt}",
            "source":      src,
            "target":      tgt,
            "relation":    e.get("keywords", e.get("relation", e.get("relationship", "related"))),
            "description": e.get("description", ""),
            "weight":      float(e.get("weight", 1.0)),
            "file_path":   e.get("file_path", ""),
            "source_ids":  [s.strip() for s in source_str.split(",") if s.strip()],
        })

    total_edges = len(edges_out)
    edges_out   = edges_out[:max_edges]
    truncated   = (total_nodes > max_nodes) or (total_edges > max_edges)

    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "meta": {
            "total_nodes":    total_nodes,
            "total_edges":    total_edges,
            "returned_nodes": len(nodes_out),
            "returned_edges": len(edges_out),
            "truncated":      truncated,
            "doc_id_filter":  doc_id,
        },
    }


# ── Health & stats ────────────────────────────────────────────────────────────
import time as _time_module
_START_TIME = _time_module.time()


@app.get("/", tags=["Health"])
def root():
    return {
        "name":    "DocSearch API",
        "version": "2.0.0",
        "docs":    "/docs",
        "status":  "ok",
        # embedding config
        "embed_model":  HF_EMBED_MODEL,
        "embed_dim":    EMBED_DIM,
        "embed_key_set": bool(HF_EMBED_API_KEY),
        # llm config
        "llm_model":   HF_LLM_MODEL,
        "llm_key_set": bool(HF_LLM_API_KEY),
        # storage paths
        "db_path":    str(DB_PATH),
        "rag_storage": str(Path("./rag_storage").resolve()),
        "upload_dir": str(UPLOAD_DIR),
        # uptime
        "uptime_s": round(_time_module.time() - _START_TIME, 1),
        # feature flags
        "features": {
            "reranker":    SENTENCE_TRANSFORMERS_OK,
            "spacy_ner":   SPACY_OK,
            "pdf_parser":  PDF_OK,
            "docx_parser": DOCX_OK,
            "llm":         LLM_OK,
        },
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":    "ok",
        "version":   "2.0.0",
        "uptime_s":  round(_time_module.time() - _START_TIME, 1),
        "embed_model":   HF_EMBED_MODEL,
        "embed_dim":     EMBED_DIM,
        "embed_key_set": bool(HF_EMBED_API_KEY),
        "llm_model":     HF_LLM_MODEL,
        "llm_key_set":   bool(HF_LLM_API_KEY),
        "db_path":       str(DB_PATH),
        "rag_storage":   str(Path("./rag_storage").resolve()),
        "upload_dir":    str(UPLOAD_DIR),
        "documents":     len(DOCS),
        "chunks":        len(CHUNKS),
        "features": {
            "reranker":    SENTENCE_TRANSFORMERS_OK,
            "spacy_ner":   SPACY_OK,
            "pdf_parser":  PDF_OK,
            "docx_parser": DOCX_OK,
            "llm":         LLM_OK,
        },
    }