# Job queue and API framework for single-machine batch ingestion

Research for [#5](https://github.com/Jarrod-Bob/acceleread/issues/5). Facts checked 2026-10-03 against official docs and GitHub (repo metadata via the GitHub API).

## Recommendation

- **Job runner: no external queue.** Build a small in-library **Job** runner: one asyncio orchestrator per Job, a `concurrent.futures.ProcessPoolExecutor` for OCR (CPU/GPU-bound), an `asyncio.Semaphore`/rate limiter around Jev HTTP calls (I/O-bound), and per-Document state persisted in **SQLite** (that table *is* the queue: status, attempts, last error). Progress = `COUNT(*) GROUP BY status`; retries = re-pick rows with `attempts < N`; resume after crash = re-pick rows not `done`.
- **Fallback if we'd rather not own that code:** **Huey with `SqliteHuey`** — the only mature, maintained queue in the list that needs no Redis/Postgres. Accept its costs (no asyncio pipeline, progress is DIY).
- **API: FastAPI.** Litestar is technically equivalent for our needs; FastAPI wins on ecosystem, maintainer activity, and the map's existing lean, and it now has built-in SSE for progress streaming.

## Constraints from the map (#1)

One machine, no distribution; 10k–100k Documents per Job; OCR local and CPU/GPU-bound; Jev cloud-only, 80 req/s and 100K tok/s; library-first, CLI and UI are clients of the API; prefer no extra infra.

## Job runners compared

| Option | Infra needed | Maintenance (GitHub, 2026-10-03) | Retries | Progress | CPU + I/O mix | Embeddable in a library |
|---|---|---|---|---|---|---|
| Celery | RabbitMQ / Redis / SQS broker (SQLAlchemy is a result *backend* only, not a broker) [1] | 5.6.3, 2026-03-26; active | Yes | Custom state via result backend | Prefork pool; no native asyncio | Heavy; separate worker process + broker |
| RQ | Redis >= 5 or Valkey >= 7.2 [2] | 2.12, 2026-08-30; active | Yes (`Retry`) | `job.meta` + `save_meta()` [3] | Fork-per-job; no Windows without WSL [2] | Needs Redis |
| arq | Redis [4] | 0.28.0, 2026-04-16; slow cadence | `Retry` exception [4] | Job status states only [4] | asyncio-native; CPU work via `run_in_executor` [4] | Needs Redis |
| Dramatiq | RabbitMQ or Redis [5] | 2.2.1, 2026-09-02; active; **LGPL** [5] | Yes (middleware) | Results middleware | Threads/processes | Needs broker; LGPL may matter for #licensing |
| Procrastinate | **PostgreSQL 13+ only** [6] | 3.10.0, 2026-09-23; active | Retry strategy [6] | Custom | sync + async workers; `run_worker_async()` [6] | Embeddable, but Postgres is extra infra |
| Huey | Redis, **SQLite**, Postgres, file, or memory [7] | 3.4.0, 2026-09-04; active | `retries`, `retry_delay`, backoff [7] | Signals + result store; no progress API [7][8] | thread / process / greenlet workers; "does not support a full asyncio pipeline" [8] | `create_consumer()` runs in-process [7]; immediate mode for tests [7] |
| litequeue (SQLite) | SQLite | ~230 stars, no releases | Manual | Manual | N/A (just a queue table) | Tiny; not worth a dependency over our own table |
| asyncio + ProcessPoolExecutor + SQLite table | None (stdlib) | Python stdlib | Ours | Ours (trivial SQL) | Native: async I/O + process pool | Fully ours |

Why the stdlib option fits best:

- **The workload is one pipeline, not heterogeneous tasks.** Every Document goes extract → (OCR pages) → classify. Generic queues model independent tasks across many workers/machines; we explicitly ruled out distribution.
- **Mixing CPU and I/O is the main problem, and asyncio solves it directly.** OCR pages go to a process pool via `loop.run_in_executor`; Jev calls stay on the event loop with a concurrency cap tuned to Jev's 80 req/s. Huey/RQ/Celery workers are thread-or-process-per-task and would either block a worker slot on HTTP or need a second queue.
- **Progress and retries are trivial when the queue is a SQLite table we own**, and the same table can serve "Record storage" (fog on the map), avoiding two stores.
- **Zero infra** keeps `pip install acceleread` + CLI usable with no daemon; the API process can host the runner.

Stdlib caveats to design around [9]:

- If a worker process dies abruptly (e.g. OOM in OCR), the pool raises `BrokenProcessPool` for all pending futures and refuses new work — the runner must recreate the pool and re-queue in-flight Documents (the SQLite state makes this safe).
- `max_tasks_per_child` (3.11+) recycles workers to contain native-library memory leaks; it forces the `spawn` start method. Python 3.14 changed the default start method away from `fork`, so OCR models load per worker process — budget startup time and RAM/VRAM per worker.
- Submitted callables must not touch Executors/Futures (deadlock).
- SQLite: use WAL mode and a single writer (the orchestrator) to avoid lock contention; workers return results instead of writing.

## API: FastAPI vs Litestar

| | FastAPI | Litestar |
|---|---|---|
| Maintenance | 0.142.2, 2026-09-30; ~103k stars; very active | 2.24.0, 2026-06-11; ~8.5k stars; active repo, slower releases |
| Progress streaming | Built-in SSE since 0.135.0: `fastapi.sse.EventSourceResponse` / `ServerSentEvent`, automatic 15 s keep-alive, `Last-Event-ID` resume [10] | Built-in `ServerSentEvent` response and `Stream` [11] |
| Background work | `BackgroundTasks` is for small in-process work; docs point to Celery-style tools for heavy compute [12] — irrelevant if our runner is in-process | Similar |
| License | MIT | MIT |

Both cover what v0 needs (JSON endpoints for submit/status/records, SSE for progress, OpenAPI for CLI/UI clients). FastAPI's larger ecosystem and the map's existing lean decide it. Pattern: the FastAPI app owns one runner instance via lifespan; `POST /jobs` enqueues rows and returns 202 + Job id; `GET /jobs/{id}` returns counts; `GET /jobs/{id}/events` streams SSE.

## Sources

1. Celery, Backends and Brokers — https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/index.html
2. RQ docs — https://python-rq.org/docs/
3. RQ jobs (`job.meta`) — https://python-rq.org/docs/jobs/
4. arq docs — https://arq-docs.helpmanual.io/
5. Dramatiq — https://dramatiq.io/
6. Procrastinate — https://procrastinate.readthedocs.io/en/stable/
7. Huey API — https://huey.readthedocs.io/en/latest/api.html
8. Huey contrib (asyncio) — https://huey.readthedocs.io/en/latest/contrib.html ; changelog https://github.com/coleifer/huey/blob/master/CHANGELOG.md
9. Python `concurrent.futures` — https://docs.python.org/3/library/concurrent.futures.html
10. FastAPI SSE — https://fastapi.tiangolo.com/tutorial/server-sent-events/
11. Litestar responses — https://docs.litestar.dev/latest/usage/responses.html
12. FastAPI Background Tasks — https://fastapi.tiangolo.com/tutorial/background-tasks/
- Release/stars/license figures: GitHub API (`latestRelease`, `stargazerCount`, `licenseInfo`) for each repo, queried 2026-10-03.

## Open questions surfaced

- **GPU OCR concurrency:** a process pool of N workers each loading a GPU model may exhaust VRAM; likely one GPU worker + N CPU workers. Depends on the OCR engine ticket.
- **Jev batching:** if Jev accepts multiple Documents per request, the I/O side is bounded by tok/s not req/s; changes the limiter design.
- **Runner hosting:** does the CLI run Jobs in-process (no server) or always talk to the API? Library-first suggests both share the same runner.
- **Cancellation / pause / resume semantics** for a Job, and how many Jobs run concurrently on one machine.
- **Record storage** (map fog): this recommendation leans toward SQLite holding both Job state and Document Records.
