# maestro-common

Shared code for the scheduler, trainers and tools (see [PLAN.md](../PLAN.md)):

| Module | What it does |
| --- | --- |
| `instances` | `Instance`: a boolean mask, a confidence and a label |
| `ls_brush` | Instances to and from Label Studio BrushLabels results (brush RLE) |
| `reward` | Per-image IoU reward: Hungarian matching of predicted and annotated instances |
| `bandit` | Beta posteriors, Thompson sampling, reshaping factor r, discounting, atomic save/load |
| `arms` | Arm names and `model_version` strings (`<family>+<strategy>@<round>`) |
| `strategies` | Uncertainty, coreset and random ranking; audit tasks; Label Studio queue scores |

Run the tests from the repo root with `uv run pytest`.
