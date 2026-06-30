"""
backend/dockling_client.py
──────────────────────────
HTTP client for the Docling document-extraction microservice.

Replaces the direct import of:
    from backend.dockling_document_extraction import chunk_elements, extract_with_dockling

Drop-in replacements with identical signatures:
    extract_with_dockling(filename: str, content: bytes) -> list[dict]
    chunk_elements(elements: list[dict], target_size: int, overlap: int) -> list[dict]

.env variables
──────────────
  DOCKLING_SERVICE_URL=http://localhost:8001
      Base URL of the running Docling microservice.
      No trailing slash.  Required — startup will warn loudly if absent.

  DOCKLING_TIMEOUT=120
      Per-request timeout in seconds (default 120).
      Large PDFs can be slow; increase if you see timeouts on big files.

  DOCKLING_MAX_RETRIES=3
      How many times to retry on transient errors (502/503/504, network drops).
      Default 3.  Set to 0 to disable retries.

  DOCKLING_RETRY_BASE_DELAY=2.0
      Base seconds for exponential back-off between retries.
      Actual delay = base * 2^(attempt-1) ± 20% jitter, capped at 30s.

Service contract (what the microservice must expose)
─────────────────────────────────────────────────────
POST /extract
    Body:  multipart/form-data  with field "file" (filename + bytes)
    Returns 200 JSON:
    {
      "elements": [
        {
          "text":           "...",          // required
          "type":           "text|table|...", // optional
          "page":           1,              // optional int
          "breadcrumb":     "Sec 1 > ...",  // optional str
          "hierarchy_path": ["Sec 1", ...], // optional list[str]
          "display_text":   "...",          // optional, falls back to text
          "data":           {...}            // optional, raw table data
        },
        ...
      ]
    }

POST /chunk
    Body:  application/json
    {
      "elements":    [...],   // same shape as /extract response elements
      "target_size": 1200,
      "overlap":     150
    }
    Returns 200 JSON:
    {
      "chunks": [
        {
          "text":              "...",     // required
          "index":             0,         // required int
          "display_text":      "...",     // optional
          "breadcrumb":        "...",     // optional
          "hierarchy_path":    [...],     // optional
          "table_part":        null,      // optional int
          "table_parts_total": null,      // optional int
          "page":              1          // optional int
        },
        ...
      ]
    }

Both endpoints should return 4xx on bad input and 5xx on internal failure.
Retries are only attempted on 502 / 503 / 504 and network errors.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, Dict, List

import aiohttp
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("docsearch.dockling_client")

# ── Configuration ─────────────────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("WARNING: %s=%r is not a valid int, using default %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("WARNING: %s=%r is not a valid float, using default %f", name, raw, default)
        return default


DOCKLING_SERVICE_URL    = os.getenv("DOCKLING_SERVICE_URL", "").rstrip("/")
DOCKLING_TIMEOUT        = _env_int(  "DOCKLING_TIMEOUT",         120)
DOCKLING_MAX_RETRIES    = _env_int(  "DOCKLING_MAX_RETRIES",       3)
DOCKLING_RETRY_BASE_DELAY = _env_float("DOCKLING_RETRY_BASE_DELAY", 2.0)

# Status codes that warrant a retry (transient service issues only)
_RETRYABLE_STATUS = {502, 503, 504}

if not DOCKLING_SERVICE_URL:
    print(
        "WARNING: DOCKLING_SERVICE_URL is not set in .env — "
        "document extraction and chunking will fail. "
        "Set it to the base URL of the Docling microservice, "
        "e.g. DOCKLING_SERVICE_URL=http://localhost:8001"
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _check_url(endpoint: str) -> str:
    """Raise immediately if the service URL is not configured."""
    if not DOCKLING_SERVICE_URL:
        raise RuntimeError(
            "DOCKLING_SERVICE_URL is not configured. "
            "Add it to .env, e.g.: DOCKLING_SERVICE_URL=http://localhost:8001"
        )
    return f"{DOCKLING_SERVICE_URL}{endpoint}"


async def _post_json(
    endpoint: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    POST JSON to a Dockling service endpoint with retry + exponential back-off.

    Returns the parsed JSON response body.
    Raises RuntimeError on all fatal errors.
    """
    url        = _check_url(endpoint)
    timeout    = aiohttp.ClientTimeout(total=DOCKLING_TIMEOUT)
    last_exc: Exception | None = None

    for attempt in range(max(1, DOCKLING_MAX_RETRIES)):
        if attempt > 0:
            base  = DOCKLING_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            jitter = base * 0.2 * (2 * random.random() - 1)
            sleep = min(base + jitter, 30.0)
            log.warning(
                "[dockling] %s attempt %d/%d — retrying in %.1fs after: %s",
                endpoint, attempt + 1, DOCKLING_MAX_RETRIES, sleep, last_exc,
            )
            await asyncio.sleep(sleep)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status in _RETRYABLE_STATUS:
                        body     = await resp.text()
                        last_exc = RuntimeError(
                            f"Dockling service transient error {resp.status} "
                            f"on {endpoint} (attempt {attempt + 1}): {body[:300]}"
                        )
                        continue

                    if resp.status >= 400:
                        body = await resp.text()
                        raise RuntimeError(
                            f"Dockling service returned HTTP {resp.status} "
                            f"on {endpoint}: {body[:400]}"
                        )

                    return await resp.json(content_type=None)

        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as exc:
            last_exc = exc
            log.warning(
                "[dockling] %s attempt %d/%d — network error: %s",
                endpoint, attempt + 1, DOCKLING_MAX_RETRIES, exc,
            )
            continue

    raise RuntimeError(
        f"Dockling service {endpoint} failed after {DOCKLING_MAX_RETRIES} attempt(s). "
        f"Last error: {last_exc}"
    ) from last_exc


async def _post_multipart(
    endpoint: str,
    filename: str,
    content: bytes,
) -> Dict[str, Any]:
    """
    POST multipart/form-data to a Dockling service endpoint.
    The file is sent as field name "file".
    """
    url        = _check_url(endpoint)
    timeout    = aiohttp.ClientTimeout(total=DOCKLING_TIMEOUT)
    last_exc: Exception | None = None

    for attempt in range(max(1, DOCKLING_MAX_RETRIES)):
        if attempt > 0:
            base   = DOCKLING_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            jitter = base * 0.2 * (2 * random.random() - 1)
            sleep  = min(base + jitter, 30.0)
            log.warning(
                "[dockling] %s attempt %d/%d — retrying in %.1fs after: %s",
                endpoint, attempt + 1, DOCKLING_MAX_RETRIES, sleep, last_exc,
            )
            await asyncio.sleep(sleep)

        try:
            form = aiohttp.FormData()
            form.add_field(
                "file",
                content,
                filename=filename,
                content_type="application/octet-stream",
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, data=form) as resp:
                    if resp.status in _RETRYABLE_STATUS:
                        body     = await resp.text()
                        last_exc = RuntimeError(
                            f"Dockling service transient error {resp.status} "
                            f"on {endpoint} (attempt {attempt + 1}): {body[:300]}"
                        )
                        continue

                    if resp.status >= 400:
                        body = await resp.text()
                        raise RuntimeError(
                            f"Dockling service returned HTTP {resp.status} "
                            f"on {endpoint}: {body[:400]}"
                        )

                    return await resp.json(content_type=None)

        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as exc:
            last_exc = exc
            log.warning(
                "[dockling] %s attempt %d/%d — network error: %s",
                endpoint, attempt + 1, DOCKLING_MAX_RETRIES, exc,
            )
            continue

    raise RuntimeError(
        f"Dockling service {endpoint} failed after {DOCKLING_MAX_RETRIES} attempt(s). "
        f"Last error: {last_exc}"
    ) from last_exc


# ── Public API — drop-in replacements ─────────────────────────────────────────

async def extract_with_dockling(filename: str, content: bytes) -> List[Dict]:
    """
    Send a file to the Dockling microservice for extraction.

    Drop-in async replacement for the synchronous
    `backend.dockling_document_extraction.extract_with_dockling`.

    Parameters
    ----------
    filename : str
        Original filename (used by the service to select the right parser).
    content  : bytes
        Raw file bytes.

    Returns
    -------
    list[dict]
        List of element dicts as returned by the service's /extract endpoint.
        Each dict has at minimum a "text" key; optional keys include
        "type", "page", "breadcrumb", "hierarchy_path", "display_text", "data".
    """
    t0 = time.perf_counter()
    log.debug("[dockling] extract: filename=%s size=%d bytes", filename, len(content))

    data = await _post_multipart("/extract", filename, content)

    elements: List[Dict] = data.get("elements", [])
    log.debug(
        "[dockling] extract: got %d elements in %.2fs",
        len(elements), time.perf_counter() - t0,
    )
    return elements


async def chunk_elements(
    elements: List[Dict],
    target_size: int = 1200,
    overlap: int = 150,
) -> List[Dict]:
    """
    Send pre-extracted elements to the Dockling microservice for chunking.

    Drop-in async replacement for the synchronous
    `backend.dockling_document_extraction.chunk_elements`.

    Parameters
    ----------
    elements    : list[dict]
        Elements as returned by extract_with_dockling().
    target_size : int
        Approximate target character size per chunk (default 1200).
    overlap     : int
        Character overlap between consecutive chunks (default 150).

    Returns
    -------
    list[dict]
        List of chunk dicts.  Each dict has at minimum:
          "text"  (str), "index" (int)
        Optional keys: "display_text", "breadcrumb", "hierarchy_path",
                       "table_part", "table_parts_total", "page".
    """
    t0 = time.perf_counter()
    log.debug(
        "[dockling] chunk: %d elements, target_size=%d, overlap=%d",
        len(elements), target_size, overlap,
    )

    data = await _post_json(
        "/chunk",
        {
            "elements":    elements,
            "target_size": target_size,
            "overlap":     overlap,
        },
    )

    chunks: List[Dict] = data.get("chunks", [])
    log.debug(
        "[dockling] chunk: got %d chunks in %.2fs",
        len(chunks), time.perf_counter() - t0,
    )
    return chunks


# ── Health probe (optional utility, e.g. for /health endpoint) ────────────────

async def check_dockling_health() -> Dict[str, Any]:
    """
    Ping the Dockling service's /health endpoint.

    Returns a dict:
      {"reachable": True,  "status": "ok", "detail": {...}}  on success
      {"reachable": False, "status": "error", "detail": str} on failure

    Does NOT retry — this is a fast probe for health-check responses.
    """
    if not DOCKLING_SERVICE_URL:
        return {
            "reachable": False,
            "status":    "error",
            "detail":    "DOCKLING_SERVICE_URL not configured",
        }

    url     = f"{DOCKLING_SERVICE_URL}/health"
    timeout = aiohttp.ClientTimeout(total=5)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    detail = await resp.json(content_type=None)
                    return {"reachable": True, "status": "ok", "detail": detail}
                return {
                    "reachable": False,
                    "status":    "error",
                    "detail":    f"HTTP {resp.status}",
                }
    except Exception as exc:
        return {"reachable": False, "status": "error", "detail": str(exc)}