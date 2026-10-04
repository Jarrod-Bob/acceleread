# SPDX-License-Identifier: Apache-2.0
"""Record the Jev cassette used by test_tracer.py. Makes one live API call.

set -a; . ./.env; set +a
uv run python packages/acceleread/tests/cassettes/record.py
"""

import asyncio
import json
from pathlib import Path

import httpx2
import typesafe_sdk as ts

from acceleread.jev import JevClassifier
from acceleread.models import JobSpec, Taxonomy
from acceleread.pipeline import run

HERE = Path(__file__).parent
FIXTURES = HERE.parent / "fixtures"


class RecordingTransport(httpx2.AsyncHTTPTransport):
    def __init__(self) -> None:
        super().__init__()
        self.exchanges: list[dict[str, object]] = []

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        response = await super().handle_async_request(request)
        body = await response.aread()
        self.exchanges.append(
            {
                "request": {
                    "method": request.method,
                    "path": request.url.path,
                    "json": json.loads(request.content),
                },
                "response": {"status": response.status_code, "json": json.loads(body)},
            }
        )
        # The body is already decoded, so drop the headers that describe the wire encoding.
        headers = [
            (k, v)
            for k, v in response.headers.items()
            if k.lower() not in {"content-encoding", "content-length", "transfer-encoding"}
        ]
        return httpx2.Response(response.status_code, headers=headers, content=body)


async def main() -> None:
    transport = RecordingTransport()
    classifier = JevClassifier(client=ts.AsyncTypeSafeClient(transport=transport))
    spec = JobSpec(
        inputs=[FIXTURES / "sample.pdf"], taxonomy=Taxonomy.from_file(FIXTURES / "taxonomy.yaml")
    )
    async for record in run(spec, classifier):
        print(record.model_dump_json(exclude={"text"}, indent=2))
    out = HERE / "jev_sector.json"
    out.write_text(json.dumps(transport.exchanges, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
