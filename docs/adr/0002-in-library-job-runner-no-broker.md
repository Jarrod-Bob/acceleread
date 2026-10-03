# In-library Job runner with a SQLite queue, no external broker

v0 runs Jobs of 10k–100k Documents on one machine. Its runner lives inside the library: an asyncio orchestrator, a `ProcessPoolExecutor` for OCR, and rate-limited async Classifier calls. Per-Document state sits in a SQLite table that doubles as the queue, so progress is a count of statuses, retries are an attempt counter, and crash-resume is built in. We chose this over Celery, RQ, arq or Dramatiq because those need Redis or RabbitMQ (Procrastinate needs Postgres). That is infrastructure a single-machine batch tool shouldn't demand. Throughput is bounded by OCR and Jev's rate limits, not by queue features.

## Considered Options

- **Huey with `SqliteHuey`:** no Redis, and maintained. It's the fallback if owning the runner proves costly. It was not chosen because it has no asyncio pipeline and progress tracking would have to be built by hand.

## Consequences

- We own crash handling, and each OCR worker's model load needs budgeting (spawn start method).
- **Amended:** extraction runs on our own small worker manager over `multiprocessing`, not `ProcessPoolExecutor`, because the executor can't kill one hung or oversized task without tearing down the whole pool. See [How many OCR workers run, and how does the pool recover from crashes?](https://github.com/Jarrod-Bob/acceleread/issues/18)
- Distributed or multi-machine workers are out of scope. Adding them later means replacing this runner, not configuring it.
- Details: [Which job queue and API framework fit single-machine batch ingestion?](https://github.com/Jarrod-Bob/acceleread/issues/5)
