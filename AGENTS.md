# AGENTS.md

Working notes for AI agents in this repository. If a rule here turns out to be wrong, fix the rule
rather than working around it.

## What this is

`nmanga` is a Python 3.13+ CLI for post-processing manga releases: renaming, splitting, leveling,
posterizing and packing, plus ML denoise/upscale through ONNX Runtime and PyTorch. Everything hangs
off one `nmanga` command tree defined in `nmanga/cmd.py`.

## Commands

```sh
uv sync --locked --extra dev            # what CI installs
uv run ruff check .                     # lint gate
uv run ruff format .                    # format
uv run pytest -q                        # tests
make test                               # tests with coverage, as CI runs them
uv run python tools/generate-schema.py  # regenerate orchestrator.jsonschema
```

`python -m py_compile <file>` every file you touch. pytest only imports what the suite reaches, so a
syntax error in a module nothing imports survives a green run.

There is no `python -m nmanga`; the entry point is the `nmanga` console script.

## Layout

| Path | What lives there |
| --- | --- |
| `nmanga/cmd.py` | The click root, the command panels and every `main.add_command` call |
| `nmanga/cli/` | One module per command or command group |
| `nmanga/ml/` | ONNX Runtime and PyTorch backends, session builders, backend names, caches |
| `nmanga/orchestrator/` | Pydantic config models and the action implementations |
| `nmanga/term/` | Console, progress and prompt helpers |
| `tests/` | Pytest suite, mirrors the module layout |
| `tools/` | Developer scripts, excluded from ruff |
| `samples/`, `_templates/` | Example orchestrator configs and release layout material |

## Conventions

- Every source file opens with the MIT header, then a one or two line comment naming the module's job.
- `from __future__ import annotations` first, type hints on every signature, an `__all__` tuple in
  modules that export anything.
- 120 columns, double quotes, four spaces. ruff's `D` docstring rules are disabled; when you write a
  docstring, keep the existing Sphinx style (`:param:`, `:return:`, `:raises:`).
- Terminal output goes through `term.get_console()`. Never `print`.
- Write the obvious version. A plain loop beats a clever comprehension, an early return beats nesting,
  and a name should say what the value is. If following one function needs two places in your head,
  simplify the function instead of adding a comment that explains it.
- Comments say why, in plain words, and stay short. No restating the line, no narrating what you are
  about to do, no jargon or flourish. A constraint a reader cannot infer from the code earns a
  comment, everything else does not.
- No decorative comment banners such as `# ---- Thingy ----------------`. If a file genuinely needs a
  section marker, write `# MARK: Thingy`, which editors fold and highlight.

## Architecture rules

**Heavy imports stay lazy.** Importing `nmanga.ml` must not pull in torch, onnxruntime, tensorrt or
windowsml. `tests/test_ml_refactor.py` enforces this with subprocesses. Import inside the function,
or under `if TYPE_CHECKING` when only annotations need the name.

**One ONNX Runtime entry point.** `nmanga.ml.ort.get_onnxruntime()` handles DLL preload, telemetry and
plugin execution provider registration, and returns a process-wide singleton. Never import
`onnxruntime` directly. `nmanga.denoiser`, `nmanga.winml` and `nmanga.lazy` are compatibility facades,
new code imports from `nmanga.ml.*`.

**Sessions come from one function.** `nmanga.ml.onnx_session.prepare_model_runtime_builders` takes an
`InferenceBackend` from `nmanga/ml/backends.py`. Raw `<Vendor>ExecutionProvider` strings must not
reach a CLI option, a config file or a pydantic model.

**Adding a CLI command** means: a module in `nmanga/cli/` with `@click.command(...)` and
`cls=NMangaCommandHandler`, an import plus `main.add_command(...)` in `nmanga/cmd.py`, and the command
name added to the matching `@click.command_panel(commands=[...])` list. Shared options live in
`nmanga/cli/options.py`, cross-cutting decorators in `nmanga/cli/_deco.py`.

**Adding an orchestrator action** means: a pydantic model under `nmanga/orchestrator/actions/`
subclassing `BaseAction`, the new kind in `ActionKind` (`_base.py`), an export from
`actions/__init__.py` (both `__all__` and the `ActionType` union), then a schema regeneration.
`extra="forbid"` is set, so removing or renaming a field breaks existing configs: call it out in the
changelog.

## Tests

- Aim for behaviour assertions that run without a GPU, a model file or TensorRT.
- The suite must pass on a machine with no CUDA runtime at all.
- On macOS, `tests/test_ml_refactor.py::test_ort_initialization_is_shared_and_runs_once` fails because
  it asserts a `preload_dlls()` call that `ml/ort.py` only makes off macOS. That is a known platform
  failure, not a regression you introduced.

## Changelog

- Newest first, one flat `## [X.Y.Z] (YYYY-MM-DD)` heading per release. Patch releases are not nested
  under minors.
- `## Unreleased` always sits at the top and holds `Nothing yet.` when nothing is pending. On a version
  bump it is filled in, dated, and a fresh empty `Unreleased` is put back above it.
- One bullet per user-visible change. Do not re-explain behaviour an earlier release already covered,
  and lead with the reason a reader cares about.
- Bump `__version__` in `nmanga/_metadata.py`. setuptools reads it dynamically, so it is the only copy.

## Docs

`README.md` covers installation and the extras matrix. Command and configuration documentation lives
in the GitHub wiki, so do not grow the README into a command reference.
