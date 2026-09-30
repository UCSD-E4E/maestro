# Maestro

Active learning for [Label Studio](https://labelstud.io/), with a multi-armed bandit that picks which model and query strategy drive labeling. See [PLAN.md](PLAN.md) for the design.

| Folder | What it is | Ships as |
| --- | --- | --- |
| [cli/](cli/) | `maestro_cli`: spins a per-developer stack up and down on Kubernetes | pipx package |
| [scheduler/](scheduler/) | Label Studio ML backend; holds the bandit and starts trainer Jobs | `ghcr.io/ucsd-e4e/maestro-scheduler:<branch>` |
| [trainer/](trainer/) | GPU Job that trains an arm and pushes predictions to Label Studio | `ghcr.io/ucsd-e4e/maestro-trainer:<branch>` |

CI builds both images on every PR to `main` and on pushes to `main`, tagged with the branch name. `maestro_cli spin up` pulls the images matching your checkout's current branch, so point both `--scheduler_path` and `--trainer_path` at this repo's root.

This repo merges the former `maestro_cli`, `maestro_scheduler` and `maestro_trainer` repos, with their full history under each folder.

## Development

Python tooling is [uv](https://docs.astral.sh/uv/). The root is a uv workspace containing `common/`, `trainer/` and `cli/`; `scheduler/` is a separate uv project until its rewrite.

```sh
uv sync --package maestro-common   # shared library only (no torch)
uv run pytest                      # common/ tests
uv sync                            # everything, including torch for trainer/
```
