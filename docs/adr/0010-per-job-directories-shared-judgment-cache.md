# Per-Job directories under a Workspace catalog; Judgment cache shared across Jobs

A Workspace holds a small `catalog.sqlite` (Job list, FIFO order, runner lease), a `cache.sqlite` (the Judgment cache), and one directory per Job. Each Job directory holds `job.sqlite` (Document state and queue, Document Records with zlib-compressed text, and the Review log), its manifest, and its copied inputs. We chose per-Job directories over one shared database because Jobs reach tens of gigabytes of text. Deleting or archiving a Job is then removing or copying one directory, rather than vacuuming a shared file, and write contention stays inside one Job. Records live in SQLite rather than as separate text files so that each Document's state and Record land in one transaction, and crash-resume can never see a half-written Record. The Judgment cache moved out of the Job (where the Classifier seam ticket first put it) into the Workspace, so a new Job that changes one Question pays only for that Question. Immutable manifests depend on that: changing a spec always means a new Job.

## Considered Options

- **One SQLite database for the whole Workspace:** simpler cross-Job queries, which nothing in v0 needs, but deleting a Job leaves a huge file to vacuum and every Job shares one writer.
- **Text in per-Document files beside the database:** a smaller database, but writes stop being atomic.

## Consequences

- One runner per Workspace, held through a heartbeat lease in the catalog (ADR 0008). A CLI run while `serve` holds the lease joins the shared FIFO instead of running its own Job.
- Querying Records across Jobs needs SQLite `ATTACH`.
- The cache grows without bound until pruned (`acceleread cache prune`). Its key includes the model version, and its format is versioned.
- SQLite WAL needs a local filesystem, so a Workspace on NFS or SMB is refused unless explicitly allowed.
- Details: [Where do Jobs, Records, uploads and Reviews live, and how long are they kept?](https://github.com/Jarrod-Bob/acceleread/issues/17)
