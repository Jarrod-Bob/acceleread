# SPDX-License-Identifier: Apache-2.0
"""Handlers that misbehave on demand, for the worker manager's tests.

Workers import them by dotted path (`worker_handlers:crash`), so they must be module-level.
"""

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

Report = Callable[[int], None]


def echo(payload: Any, report: Report) -> Any:
    return payload


def pid(payload: Any, report: Report) -> int:
    time.sleep(payload or 0)
    return os.getpid()


def omp_thread_limit(payload: Any, report: Report) -> str | None:
    return os.environ.get("OMP_THREAD_LIMIT")


def crash(payload: Any, report: Report) -> None:
    os._exit(7)


def crash_once(payload: Any, report: Report) -> int:
    """Crash the first time it runs (marker file absent), succeed on the retry."""
    marker = Path(payload)
    if not marker.exists():
        marker.write_text("crashed")
        os._exit(7)
    return os.getpid()


def hang(payload: Any, report: Report) -> None:
    time.sleep(3600)


def sleep_after_report(payload: Any, report: Report) -> str:
    """payload = (ocr_pages, seconds): announce OCR Pages, then work for a while."""
    ocr_pages, seconds = payload
    report(ocr_pages)
    time.sleep(seconds)
    return "done"


def hog(payload: Any, report: Report) -> None:
    """Hold `payload` megabytes, then wait to be killed."""
    held = bytearray(int(payload) * 1024 * 1024)
    for i in range(0, len(held), 4096):
        held[i] = 1
    time.sleep(3600)


def fail(payload: Any, report: Report) -> None:
    raise ValueError("this Document is broken")
