"""
Thread-safe log queue for routing tool progress messages to the Streamlit UI.

Usage:
  - Call set_active_queue(q) at the start of a request with a fresh queue.Queue.
  - Call emit(msg) from anywhere (any thread) to send a log line.
  - Call clear_active_queue() once the request is done.
"""
import queue
import threading

_lock = threading.Lock()
_active_queue: "queue.Queue | None" = None


def set_active_queue(q: "queue.Queue") -> None:
    global _active_queue
    with _lock:
        _active_queue = q


def clear_active_queue() -> None:
    global _active_queue
    with _lock:
        _active_queue = None


def emit(message: str) -> None:
    """Send a log line to the active UI queue (if any) and print to terminal."""
    print(message)
    with _lock:
        if _active_queue is not None:
            _active_queue.put_nowait(("_LOG", message))
