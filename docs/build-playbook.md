# v0 build playbook

How to build acceleread v0 from the spec: one fresh subagent per issue, each working test-first, coordinated by an orchestrating session. Read this at the start of every build session.

## Where everything is

| What | Where |
|---|---|
| Spec (the source of truth) | [`docs/spec/v0.md`](spec/v0.md), §1–§15 |
| Decisions and their reasoning | [`docs/adr/`](adr/) 0001–0010; the closed [wayfinder map](https://github.com/Jarrod-Bob/acceleread/issues/1) and its tickets |
| Glossary | [`CONTEXT.md`](../CONTEXT.md). Use its terms exactly (Document, Page, Section, Judgment, Workspace, …) |
| Build issues | Parent [acceleread v0 build](https://github.com/Jarrod-Bob/acceleread/issues/27), milestone `v0`, labels `area:*`. Order uses GitHub native blocking |
| Dogfood corpus | `corpus/` (filings in gitignored `corpus/data/`; rebuild with `corpus/fetch.py`) |
| Jev API key | `TYPESAFE_API_KEY` in `.env` (gitignored). Load with `set -a; . ./.env; set +a` |

## State when this was written (2026-10-04)

- **Merged:**
  - [Restructure into a uv workspace with CI](https://github.com/Jarrod-Bob/acceleread/issues/28) (PR #50).
  - [Tracer: PDF → Record through Jev via `acceleread run`](https://github.com/Jarrod-Bob/acceleread/issues/29) (PR #51). It was written test-after, without the TDD skill. Everything from here on is test-first.
- **Frontier:** these four can run in parallel:
  - [Domain models and `jobs:validate`](https://github.com/Jarrod-Bob/acceleread/issues/30)
  - [Workspace storage](https://github.com/Jarrod-Bob/acceleread/issues/31)
  - [Extraction workers and the OCR rule](https://github.com/Jarrod-Bob/acceleread/issues/32)
  - [Classifier seam, Jev adapter and rate limiter](https://github.com/Jarrod-Bob/acceleread/issues/33)

  [File the Docling torch-free fixes upstream](https://github.com/Jarrod-Bob/acceleread/issues/49) is unblocked at any time, but it posts to someone else's repo, so ask the user first. (Examples is blocked by the runner, not by the tracer.)

To find the frontier at any time: open sub-issues of #27 that are unassigned and whose blockers are all closed.

```sh
for n in $(gh api repos/Jarrod-Bob/acceleread/issues/27/sub_issues --paginate --jq '.[]|select(.state=="open")|.number'); do
  open=$(gh api repos/Jarrod-Bob/acceleread/issues/$n/dependencies/blocked_by --jq '[.[]|select(.state=="open")]|length')
  who=$(gh issue view $n --json assignees --jq '[.assignees[].login]|join(",")')
  [ "$open" = 0 ] && [ -z "$who" ] && gh issue view $n --json number,title --jq '"\(.number)\t\(.title)"'
done
```

## The loop (orchestrating session)

1. **Merge what the user approved**, pull `main`, and recompute the frontier.
2. **Pick issues for this round.** Run several frontier issues in parallel only when they touch different modules (see "Avoiding conflicts"). Otherwise sequence them.
3. **Claim each issue** (`gh issue edit N --add-assignee @me`) before launching its subagent.
4. **Launch one subagent per issue** with the Agent tool, using the prompt template below. Set `isolation: "worktree"` and **`model: "sonnet"`**: the spec and issues are detailed enough that Sonnet builds from them, which saves usage. Keep design judgment in the orchestrating session. Launch independent issues in the same message so they run in parallel.
5. **Review each PR when its subagent reports:**
   - Run the **`mattpocock-skills:code-review`** skill on the PR's branch against its merge-base with `main`. It reviews two axes in parallel: Standards (this repo's conventions, from this playbook) and Spec (the issue and `docs/spec/v0.md`).
   - Then work through the review checklist below.
   - Send findings back to the **same** subagent with SendMessage, so it fixes them test-first. Don't start a new subagent for fixes.
   - Re-run the review if the fixes were substantial.
6. **Report to the user**: each PR with a one-paragraph summary, what the code review found and how it was resolved, and anything that needs their decision. **The user decides merges**, unless they've said otherwise in this session.
7. **Repeat.** When an issue reveals something a later issue needs, comment it on that later issue, as was done for the planner after the tracer.

Stop and ask the user, without guessing, when:
- an issue conflicts with the spec or an ADR;
- a dependency fails the licence check;
- the spec is silent on something user-visible;
- an action posts outside this repo.

Record each decision where it belongs: the ticket, an ADR amendment, or the spec. Do it in the same PR.

## Subagent prompt template

Fill in `{N}`, `{TITLE}` and `{SLUG}`. Keep the rest verbatim.

```text
You are implementing one build issue of acceleread v0, test-first.

Repo: /Users/jarrodng/acceleread (a uv workspace; you are in your own git worktree).
Issue: #{N} "{TITLE}". Read it first: gh issue view {N} --comments
Work on branch build/{SLUG}, created from the latest origin/main.

Before writing code:
1. Load the TDD skill: call the Skill tool with skill "mattpocock-skills:tdd". Follow its
   red-green-refactor loop for every behaviour in the issue checklist.
2. Read docs/build-playbook.md (conventions), the spec sections the issue links
   (docs/spec/v0.md), the ADRs it links (docs/adr/), and CONTEXT.md for terms.
3. Read the existing code in packages/acceleread/src/acceleread/ and its tests, and build on them.
   Don't rewrite modules other issues own (see "Avoiding conflicts" in the playbook).

Rules:
- The spec and ADRs are decided. If the issue, spec and code disagree, or the spec is silent
  on something user-visible, stop and report the question. Don't choose silently.
- Tests never touch the network. Jev calls replay recorded cassettes through the real SDK via
  httpx2.MockTransport (see packages/acceleread/tests/test_tracer.py and
  tests/cassettes/record.py). A live test may exist behind ACCELEREAD_LIVE_JEV=1.
- Before opening the PR, all of these must pass:
    uv run pytest
    uv run ruff check . && uv run ruff format --check .
    uv run mypy
    uv run python tools/check_spdx.py
  If you add a dependency: uv lock, then run tools/check_licenses.py in an environment synced
  with that extra (see the playbook). Never add a dependency outside the ADR 0003 allowlist
  without stopping to ask.
- Commit messages and the PR body end with the attribution lines in the playbook.
- Open a PR against main titled "{TITLE}" with "Closes #{N}" in the body. Its body ticks off
  the issue checklist, lists anything deferred, and names any finding later issues need.
  Do not merge.

Report back: PR URL, what was built, the tests added, anything deferred, and every question or
spec conflict you hit.
```

## Review checklist (orchestrator, before reporting a PR)

- [ ] `mattpocock-skills:code-review` has run on the branch, and its Standards and Spec findings are fixed or explicitly accepted with a reason in the PR.
- [ ] Every box in the issue checklist is either done or explicitly deferred with a reason.
- [ ] Behaviour matches the linked spec sections and ADRs, and names match `CONTEXT.md` and the spec's schema (§6).
- [ ] Tests describe behaviour through public interfaces. They don't mirror the implementation, and they don't use the network.
- [ ] CI is green on the PR (`gh pr checks N`), including the licence jobs.
- [ ] No silent spec changes. Any amendment is in the same PR and was decided by the user.
- [ ] Findings for later issues are commented on those issues.

## Conventions

- **Layout:**
  - `packages/acceleread/src/acceleread/`: the library.
  - `packages/acceleread/tests/`: its tests, with fixtures and cassettes.
  - `packages/acceleread-eval/`: the evaluation harness.
  - `examples/*`: examples, each a workspace member with its own `pyproject.toml`.
  - `tools/`: repo checks.
  - `corpus/`: research scripts, not linted.
- **Commands:**
  - `uv sync --all-packages` to set up.
  - `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy` (strict), `uv run python tools/check_spdx.py`.
- **Licence check for one extra:**

  ```sh
  UV_PROJECT_ENVIRONMENT=/tmp/acc-lic uv sync --no-dev --package acceleread --extra quality
  UV_PROJECT_ENVIRONMENT=/tmp/acc-lic uv run --no-sync python tools/check_licenses.py
  ```

  Reviewed exceptions are in `tools/license-exceptions.toml`: `cysignals` (LGPL, narrow rule) and `unidecode` (GPL, `[edgar]` only, never in an image).
- **Every Python file** starts with `# SPDX-License-Identifier: Apache-2.0`.
- **Async tests** use pytest-asyncio in auto mode.
- **Records always come out,** even on failure (`failed` or `partial`, with `errors[]`). Never log Document text, Classifier state or secrets.
- **Branches** are `build/<slug>`. Squash-merge, and delete the branch.
- **Attribution.** Commit messages end with:

  ```text
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```

  PR bodies end with:

  ```text
  🤖 Generated with [Claude Code](https://claude.com/claude-code)
  ```

  Use the newest attribution lines the session's system prompt provides, if they differ.

## Avoiding conflicts between parallel issues

The tracer left a small codebase that several issues extend. Each issue owns these files:

| Issue | Owns | May touch lightly |
|---|---|---|
| Domain models and `jobs:validate` | `models.py`, new `validate.py`, Question Set loading | the `JobSpec` and `DocumentRecord` fields it adds |
| Workspace storage | new `workspace/` package | — |
| Extraction workers and the OCR rule | `extract.py`, new `workers.py`, new `ocr_rule.py` | `Page` provenance fields in `models.py` |
| Classifier seam, Jev adapter and rate limiter | `classifier.py`, `jev.py`, new `ratelimit.py` | — |
| Planner | `pipeline.py` planning, which moves to a new `planner.py` | — |

If two parallel PRs both touch `models.py`, merge the models issue first and rebase the other.

## Findings already recorded for later issues

- **Planner** ([#35](https://github.com/Jarrod-Bob/acceleread/issues/35)):
  - Add a fixed per-request overhead to the token estimate (286 estimated vs 506 actual on a 2-page Document).
  - Coverage should record character spans, not only `page_ranges`.
- **Release** ([#46](https://github.com/Jarrod-Bob/acceleread/issues/46)):
  - The default image has no `[edgar]`.
  - The `cysignals` LGPL text ships in `THIRD_PARTY_NOTICES.md`.
  - edgartools plans to drop `unidecode` ([dgunning/edgartools#934](https://github.com/dgunning/edgartools/issues/934)); revisit when it lands.
