"""
lazy_converter.py
─────────────────
Thread-safe, import-lazy singleton loader for Docling's DocumentConverter.

Three levels of laziness
─────────────────────────
1. Import-time  – `docling` is not imported at module load; only when the
                  converter is first requested (or pre-warmed).
2. Call-time    – `_get_converter()` builds the instance on first call, then
                  caches it forever.
3. Background   – `preload_converter()` starts a daemon thread so the heavy
                  model-weight loading happens in the background while your
                  app does other work (e.g. reading the file, building a
                  queue).  Subsequent `_get_converter()` calls block until
                  the thread finishes — typically zero wait if enough time
                  has passed.

Usage
─────
# Option A – purely on-demand (original behaviour, but now truly lazy):
    from lazy_converter import _get_converter
    converter = _get_converter()        # blocks here on first call only

# Option B – background pre-warm at app startup:
    from lazy_converter import preload_converter, _get_converter
    preload_converter()                 # fire-and-forget; returns immediately
    ... (do file I/O, argument parsing, etc.) ...
    converter = _get_converter()        # likely zero wait by now

# Option C – explicit async-style check:
    from lazy_converter import is_converter_ready, _get_converter
    if is_converter_ready():
        converter = _get_converter()    # guaranteed no wait
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only for type checkers; never executed at runtime until _load() runs.
    from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

# ── Internal state ────────────────────────────────────────────────────────────

_converter: "DocumentConverter | None" = None
_lock = threading.Lock()
_loading_thread: threading.Thread | None = None
_load_error: BaseException | None = None


# ── Core loader ───────────────────────────────────────────────────────────────

def _load() -> None:
    """
    Import docling and instantiate DocumentConverter.
    Safe to call from any thread; protected by _lock at the call sites.
    Captures any exception so callers can re-raise it on the main thread.
    """
    global _converter, _load_error
    try:
        logger.debug("lazy_converter: importing docling …")
        from docling.document_converter import DocumentConverter  # noqa: PLC0415
        logger.debug("lazy_converter: instantiating DocumentConverter …")
        _converter = DocumentConverter()
        logger.debug("lazy_converter: DocumentConverter ready.")
    except BaseException as exc:  # noqa: BLE001
        _load_error = exc
        logger.error("lazy_converter: failed to load DocumentConverter: %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def preload_converter() -> threading.Thread:
    """
    Start loading DocumentConverter in a background daemon thread.

    Safe to call multiple times — subsequent calls are no-ops once loading
    has started (or finished).  Returns the thread object so callers can
    join() it explicitly if desired.

    Example
    -------
        from lazy_converter import preload_converter
        preload_converter()   # called at app startup; returns immediately
    """
    global _loading_thread

    with _lock:
        if _converter is not None or _loading_thread is not None:
            # Already done or already in flight — nothing to do.
            return _loading_thread  # type: ignore[return-value]

        def _bg_load():
            with _lock:
                if _converter is None and _load_error is None:
                    _load()

        thread = threading.Thread(target=_bg_load, name="docling-preload", daemon=True)
        _loading_thread = thread

    thread.start()
    logger.debug("lazy_converter: background pre-warm thread started.")
    return thread


def _get_converter() -> "DocumentConverter":
    """
    Return the shared DocumentConverter, loading it on first call.

    Thread-safe: concurrent callers block until loading completes; only one
    actually calls _load().

    Raises
    ------
    RuntimeError
        If a previous loading attempt failed (wraps the original exception).
    """
    global _converter, _loading_thread

    # Fast path — already ready, no lock needed (reading a reference is atomic
    # in CPython, and we only ever assign it once under the lock).
    if _converter is not None:
        return _converter

    with _lock:
        # Re-check inside the lock (another thread may have just finished).
        if _converter is not None:
            return _converter

        if _load_error is not None:
            raise RuntimeError(
                "DocumentConverter could not be loaded."
            ) from _load_error

        # If a background thread is in flight we need to wait for it, but it
        # also holds _lock when it calls _load(), so we can't just join() here
        # without deadlocking.  Instead, release the lock and spin-wait briefly.
        if _loading_thread is not None and _loading_thread.is_alive():
            pass  # fall through — _lock release lets the bg thread finish

    # Background thread is running — wait for it outside the lock.
    if _loading_thread is not None:
        logger.debug("lazy_converter: waiting for background pre-warm thread …")
        _loading_thread.join()

    # Now (re-)acquire the lock for a final check / cold load.
    with _lock:
        if _converter is not None:
            return _converter
        if _load_error is not None:
            raise RuntimeError(
                "DocumentConverter could not be loaded."
            ) from _load_error
        # Neither pre-warm nor previous call loaded it — do it now.
        _load()
        if _load_error is not None:
            raise RuntimeError(
                "DocumentConverter could not be loaded."
            ) from _load_error

    return _converter  # type: ignore[return-value]


def is_converter_ready() -> bool:
    """Return True if the converter is already loaded (non-blocking check)."""
    return _converter is not None


def reset_converter() -> None:
    """
    Drop the cached converter and allow re-instantiation.

    Intended for testing or hot-reload scenarios only.  Not safe to call
    while a conversion is in progress.
    """
    global _converter, _loading_thread, _load_error
    with _lock:
        _converter      = None
        _loading_thread = None
        _load_error     = None
    logger.debug("lazy_converter: converter cache cleared.")