# acceleread

Accelerated document ingestion: OCR PDFs and images, then classify them cheaply with System One models (e.g. TypeSafe Jev) instead of an LLM. Usable standalone or via an API.

> Status: building v0. See the [spec](docs/spec/v0.md) and the [v0 milestone](https://github.com/Jarrod-Bob/acceleread/milestone/1).

## Development

This repo is a [uv](https://docs.astral.sh/uv/) workspace: the library is in `packages/acceleread`, the evaluation harness in `packages/acceleread-eval`, and examples in `examples/`.

```sh
uv sync --all-packages        # create .venv with every workspace member
uv run pytest                 # tests
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run python tools/check_spdx.py
uv run --no-sync python tools/check_licenses.py   # licences of the synced environment (ADR 0003)
```

Licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
