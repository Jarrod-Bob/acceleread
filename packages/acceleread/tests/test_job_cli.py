# SPDX-License-Identifier: Apache-2.0
"""The CLI's Job commands (docs/spec/v0.md §9): run, validate, status, jobs, cancel, resume,
retry --failed, records --flagged, export, and `--server URL`."""

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import httpx
import pytest

from acceleread.classifier import ClassifierUnavailable
from acceleread.cli import main
from acceleread.jobcontrol import cancel_job
from acceleread.models import JobSpec, Taxonomy
from acceleread.ratelimit import RateLimit, RateLimitedClassifier
from acceleread.runner import Runner
from acceleread.serverclient import ServerClient
from acceleread.workspace import Workspace

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.pdf"
FILING = FIXTURES / "filing10k.pdf"
TAXONOMY_FILE = FIXTURES / "taxonomy.yaml"
Fake = Callable[..., Any]


class Cli:
    def __init__(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self.workspace = tmp_path / "ws"
        self.capsys = capsys

    def __call__(self, *args: str) -> tuple[int, str, str]:
        code = main(["--workspace", str(self.workspace), *args])
        captured = self.capsys.readouterr()
        return code, captured.out, captured.err

    def open(self) -> Workspace:
        return Workspace.open(self.workspace)


@pytest.fixture
def cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Cli:
    return Cli(tmp_path, capsys)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, fake_classifier: Fake) -> Any:
    classifier = fake_classifier()
    monkeypatch.setattr("acceleread.cli.make_classifier", lambda model: classifier)
    return classifier


def lines(out: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.splitlines()]


def test_run_takes_flags_and_writes_jsonl(cli: Cli, fake: Any, tmp_path: Path):
    code, out, err = cli(
        "run", str(SAMPLE), str(FILING), "--taxonomy", str(TAXONOMY_FILE), "--no-cache"
    )
    assert code == 0
    first, second = lines(out)
    assert first["classification"]["value"] == "energy" and second["status"] == "ok"
    assert "sample.pdf: ok" in err  # progress goes to stderr
    with cli.open() as ws:
        (job,) = ws.list_jobs()
        manifest = json.loads((ws.job_dir(job.id) / "manifest.json").read_text())
    assert manifest["cache"] is False


def test_run_reads_a_job_yaml_and_flags_override_it(cli: Cli, fake: Any, tmp_path: Path):
    job = tmp_path / "job.yaml"
    job.write_text(
        f"inputs: [{SAMPLE}]\nmax_cost_usd: 5\ntaxonomy:\n  name: t\n  categories:\n"
        "    - {name: energy}\n"
    )
    code, out, _ = cli("run", "--job", str(job), "--max-cost-usd", "9")
    assert code == 0 and lines(out)[0]["classification"]["value"] == "energy"
    with cli.open() as ws:
        (info,) = ws.list_jobs()
        manifest = json.loads((ws.job_dir(info.id) / "manifest.json").read_text())
    assert manifest["max_cost_usd"] == 9


def test_run_with_an_invalid_spec_exits_2_and_says_why(cli: Cli, fake: Any):
    code, out, err = cli(
        "run", str(SAMPLE), "--taxonomy", str(TAXONOMY_FILE), "--ocr-language", "de"
    )
    assert code == 2 and out == ""
    assert "add-language de" in err


def test_run_without_inputs_is_a_usage_error(cli: Cli, fake: Any):
    code, _, err = cli("run", "--taxonomy", str(TAXONOMY_FILE))
    assert code == 2 and "no inputs" in err


def test_validate_reports_errors_and_warnings_without_running(cli: Cli, fake: Any):
    code, _, err = cli("validate", str(SAMPLE), "--taxonomy", str(TAXONOMY_FILE))
    assert code == 0 and fake.calls == []
    code, _, err = cli(
        "validate", str(SAMPLE), "--taxonomy", str(TAXONOMY_FILE), "--ocr-language", "de"
    )
    assert code == 2 and "unsupported_ocr_language" in err
    code, _, err = cli("validate", str(SAMPLE))
    assert code == 2 and "no_judgments" in err


def test_validate_sees_language_packs_installed_in_the_workspace(cli: Cli, fake: Any):
    pack = cli.workspace / "models" / "tessdata"
    pack.mkdir(parents=True)
    (pack / "deu.traineddata").write_bytes(b"pack")
    code, _, _ = cli(
        "validate", str(SAMPLE), "--taxonomy", str(TAXONOMY_FILE), "--ocr-language", "de"
    )
    assert code == 0


def test_jobs_status_and_export(cli: Cli, fake: Any, tmp_path: Path):
    assert cli("run", str(SAMPLE), "--taxonomy", str(TAXONOMY_FILE))[0] == 0
    with cli.open() as ws:
        (job,) = ws.list_jobs()

    _, out, _ = cli("jobs")
    assert job.id in out and "done" in out and "1 docs" in out

    _, out, _ = cli("status")  # the latest Job by default
    assert f"Job {job.id}" in out and "1 done" in out and "estimated cost" in out
    _, out, _ = cli("status", job.id, "--json")
    assert json.loads(out)["progress"]["documents"] == 1

    shown = tmp_path / "export.jsonl"
    code, _, _ = cli("export", job.id, "--include-text", "false", "-o", str(shown))
    (record,) = lines(shown.read_text())
    assert code == 0 and "text" not in record and record["pages"]
    manifest = json.loads((tmp_path / "export.manifest.json").read_text())  # beside the JSONL
    assert manifest["taxonomy"]["name"] == "sector"

    _, out, _ = cli("export", job.id)
    assert "text" in lines(out)[0]


def test_status_of_an_unknown_job_fails(cli: Cli, fake: Any):
    cli("jobs")
    code, _, err = cli("status", "nope")
    assert code == 1 and "nope" in err


def test_cancel_then_resume_finishes_the_job(cli: Cli, fake: Any, tmp_path: Path):
    with cli.open() as ws:
        job_id = Runner(ws, fake).submit(
            JobSpec(inputs=[SAMPLE], taxonomy=Taxonomy.from_file(TAXONOMY_FILE))
        )
    code, out, _ = cli("cancel", job_id)
    assert code == 0 and f"cancelled {job_id}" in out
    assert cli("cancel", job_id)[0] == 1  # already finished

    code, out, err = cli("resume", job_id)
    assert code == 0 and "re-queued 1 Documents" in err
    code, out, _ = cli("export", job_id)
    assert lines(out)[0]["status"] == "ok"
    assert cli("resume", job_id)[1].startswith("nothing to resume")


def test_retry_failed_requeues_and_reruns_failed_documents(
    cli: Cli, fake_classifier: Fake, monkeypatch: pytest.MonkeyPatch
):
    broken = fake_classifier(raises=lambda _: ClassifierUnavailable("503"))
    with cli.open() as ws:
        runner = Runner(
            ws, RateLimitedClassifier(broken, RateLimit(1e9, 1e6), max_unavailable_tries=1)
        )
        job_id = runner.submit(JobSpec(inputs=[SAMPLE], taxonomy=Taxonomy.from_file(TAXONOMY_FILE)))
        import asyncio

        asyncio.run(runner.execute(job_id))
    monkeypatch.setattr("acceleread.cli.make_classifier", lambda model: fake_classifier())

    with pytest.raises(SystemExit):  # `--failed` is required: the only mode in v0
        cli("retry", job_id)
    code, _, err = cli("retry", job_id, "--failed")
    assert code == 0 and "re-queued 1 Documents" in err
    record = lines(cli("export", job_id)[1])[0]
    assert record["status"] == "ok" and record["attempts"] == 2


def test_records_flagged_lists_only_judgments_flagged_for_review(cli: Cli, fake: Any):
    assert cli("run", str(SAMPLE), str(FILING), "--taxonomy", str(TAXONOMY_FILE))[0] == 0
    with cli.open() as ws:
        (job,) = ws.list_jobs()
        with ws.open_job(job.id) as store:
            record = store.get_record("000001")
            assert record is not None and record.classification is not None
            record.classification.escalation.status = "flagged"
            store.save_record("000001", "done", record)

    code, out, _ = cli("records", job.id, "--flagged")
    assert code == 0
    (flagged,) = lines(out)
    assert flagged["source"]["filename"] == "filing10k.pdf"
    assert len(lines(cli("records", job.id)[1])) == 2
    assert "text" not in lines(cli("records", job.id, "--include-text", "false")[1])[0]


# --server


def served(monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "acceleread.jobcli.ServerClient", partial(ServerClient, transport=transport)
    )


def test_server_commands_go_through_the_api(cli: Cli, monkeypatch: pytest.MonkeyPatch, fake: Any):
    seen: list[str] = []
    summary = {
        "job_id": "j1",
        "kind": "job",
        "state": "running",
        "progress": {
            "documents": 4,
            "by_state": {"done": 1, "queued": 3},
            "eta_seconds": None,
            "throughput_per_s": 1.0,
        },
        "classifier": {
            "model": "jev",
            "requests": 1,
            "input_tokens": 10,
            "cache_hits": 0,
            "estimated_cost_usd": 0.0,
            "priced": True,
        },
        "extraction": {"pages": 1, "pages_ocr": 0},
        "failures": {},
        "flags": {"stalled": False, "cancel_reason": None},
        "max_cost_usd": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        path = request.url.path
        if path == "/v1/jobs":
            return httpx.Response(
                200,
                json=[
                    {"id": "j1", "kind": "job", "state": "running", "created_at": 0, "documents": 4}
                ],
            )
        if path == "/v1/jobs/j1/export":
            assert request.url.params["include_text"] == "false"
            return httpx.Response(200, text='{"a": 1}\n{"a": 2}\n')
        if path == "/v1/jobs/j1":
            return httpx.Response(200, json=summary)
        if path == "/v1/jobs/missing/cancel":
            return httpx.Response(404, json={"detail": "no such Job"})
        return httpx.Response(204)

    served(monkeypatch, handler)
    assert "j1" in cli("--server", "http://x", "jobs")[1]
    assert "Job j1" in cli("--server", "http://x", "status", "j1")[1]
    assert "Job j1" in cli("--server", "http://x", "status")[1]
    assert (
        cli("--server", "http://x", "export", "j1", "--include-text", "false")[1]
        == '{"a": 1}\n{"a": 2}\n'
    )
    assert cli("--server", "http://x", "cancel", "j1")[0] == 0
    assert cli("--server", "http://x", "resume", "j1")[0] == 0
    assert cli("--server", "http://x", "retry", "j1", "--failed")[0] == 0
    code, _, err = cli("--server", "http://x", "cancel", "missing")
    assert code == 1 and "no such Job" in err
    assert "POST /v1/jobs/j1/cancel" in seen and "POST /v1/jobs/j1/retry" in seen


def test_commands_the_api_cannot_serve_yet_say_so(cli: Cli, fake: Any):
    code, _, err = cli("--server", "http://x", "run", str(SAMPLE), "--taxonomy", str(TAXONOMY_FILE))
    assert code == 2 and "--server" in err


def test_cancel_marks_a_running_job_for_its_runner(cli: Cli, fake: Any):
    with cli.open() as ws:
        runner = Runner(ws, fake, holder="live")
        job_id = runner.submit(JobSpec(inputs=[SAMPLE], taxonomy=Taxonomy.from_file(TAXONOMY_FILE)))
        assert runner.acquire()
        ws.set_job_state(job_id, "running")
        assert cancel_job(ws, job_id) == "requested"
    code, out, _ = cli("cancel", job_id)
    assert code == 0 and "requested" in out


def test_run_exits_1_only_for_failed_documents(
    cli: Cli, fake_classifier: Fake, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # A Classifier error leaves a `partial` Record: the run still succeeded.
    flaky = fake_classifier(raises=lambda _: RuntimeError("boom"))
    monkeypatch.setattr("acceleread.cli.make_classifier", lambda model: flaky)
    code, out, _ = cli("run", str(SAMPLE), "--taxonomy", str(TAXONOMY_FILE))
    assert lines(out)[0]["status"] == "partial" and code == 0

    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    code, out, _ = cli("run", str(broken), "--taxonomy", str(TAXONOMY_FILE))
    assert lines(out)[0]["status"] == "failed" and code == 1


def test_run_writes_the_manifest_beside_its_output(cli: Cli, fake: Any, tmp_path: Path):
    out = tmp_path / "records.jsonl"
    assert cli("run", str(SAMPLE), "--taxonomy", str(TAXONOMY_FILE), "-o", str(out))[0] == 0
    manifest = json.loads((tmp_path / "records.manifest.json").read_text())
    assert manifest["taxonomy"]["name"] == "sector"


def test_export_over_the_server_fetches_the_manifest_too(
    cli: Cli, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/manifest"):
            return httpx.Response(200, text='{"model": "jev"}')
        return httpx.Response(200, text='{"a": 1}\n')

    served(monkeypatch, handler)
    out = tmp_path / "x.jsonl"
    assert cli("--server", "http://x", "export", "j1", "-o", str(out))[0] == 0
    assert out.read_text() == '{"a": 1}\n'
    assert (tmp_path / "x.manifest.json").read_text() == '{"model": "jev"}'


def test_validate_prints_the_cost_and_duration_estimates(cli: Cli, fake: Any):
    code, out, _ = cli("validate", str(SAMPLE), "--taxonomy", str(TAXONOMY_FILE))
    assert code == 0
    assert "estimated cost up to $" in out and "Classifier-bound duration" in out
