# SPDX-License-Identifier: Apache-2.0
"""A small client for the `/v1` HTTP API (docs/spec/v0.md §9), behind the CLI's `--server URL`.

The API itself (`serve`) is a later build issue (#41), which fixes the response shapes. This
client assumes the obvious ones: `GET /v1/jobs` returns a list of Jobs, and `GET /v1/jobs/{id}`
returns the Job summary (§11), the same dict `acceleread status` prints in-process.
"""

import os
from collections.abc import Iterator
from types import TracebackType
from typing import Any, Self

import httpx


class ServerError(RuntimeError):
    """The server refused a request or could not be reached."""


class ServerClient:
    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        token = token if token is not None else os.environ.get("ACCELEREAD_API_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/v1", headers=headers, transport=transport, timeout=60
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _check(self, response: httpx.Response) -> httpx.Response:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ServerError(f"{response.request.method} {response.request.url.path}: {detail}")
        return response

    def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            return self._check(self._http.request(method, path, **kw))
        except httpx.TransportError as err:
            raise ServerError(f"cannot reach the server: {err}") from err

    def jobs(self, *, include_ingest: bool = False) -> list[dict[str, Any]]:
        params = {"kind": "ingest"} if include_ingest else None
        jobs: list[dict[str, Any]] = self._request("GET", "/jobs", params=params).json()
        return jobs

    def summary(self, job_id: str) -> dict[str, Any]:
        summary: dict[str, Any] = self._request("GET", f"/jobs/{job_id}").json()
        return summary

    def cancel(self, job_id: str) -> None:
        self._request("POST", f"/jobs/{job_id}/cancel")

    def resume(self, job_id: str) -> None:
        self._request("POST", f"/jobs/{job_id}/resume")

    def retry(self, job_id: str) -> None:
        self._request("POST", f"/jobs/{job_id}/retry")

    def manifest(self, job_id: str) -> str:
        """The Job's resolved manifest, as the server serves it."""
        return self._request("GET", f"/jobs/{job_id}/manifest").text

    def export(self, job_id: str, *, include_text: bool = True) -> Iterator[str]:
        """The Job's JSONL, one line per Document, streamed."""
        params = {"include_text": "true" if include_text else "false"}
        try:
            with self._http.stream("GET", f"/jobs/{job_id}/export", params=params) as response:
                if response.status_code >= 400:
                    response.read()
                    self._check(response)
                yield from response.iter_lines()
        except httpx.TransportError as err:
            raise ServerError(f"cannot reach the server: {err}") from err
