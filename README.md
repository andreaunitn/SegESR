# SegESR repository

Research on real-world image super-resolution (Real-ISR). The original SegESR method and the directions tried after it
are closed. Their code is in `archive/`, and their models and outputs on the cluster are in `preset/archive/`.

- [docs/ROADMAP.md](docs/ROADMAP.md): the directions, their steps and the decisions taken at each gate.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): every job with its commit, flags and metrics.
- [docs/ARCHIVE.md](docs/ARCHIVE.md): where the code, models and results of each closed direction are.

## Active code

| Path | Content |
|---|---|
| `data_tools/metrics.py` | Full-reference and no-reference metrics of an SR output folder (`metrics/<dataset>/results_<name>.json`) |
| `data_tools/fine_structure.py` | Per-band PSNR and energy, edge metrics (`metrics/<dataset>/finestruct_<name>.json`) |
| `data_tools/results_table.py` | One table per dataset from those files (standard library only) |
| `data_tools/collect_sr.py` | Brings the outputs of external models to the layout the metrics read |
| `data_tools/download_data.py` | Test and training datasets, base models |
| `cluster/onestep_baselines.sh` | PiSA-SR and OSEDiff (official or other weights): generation and metrics on RealSR, DRealSR, DIV2K |
| `cluster/fine_structure.sh` | Fine-structure metrics of existing outputs |
| `cluster/setup_env.sh`, `cluster/setup_bibandsr.sh` | The two environments: the container venv (metrics, and SeeSR) and the PiSA-SR and OSEDiff venv |
| `cluster/env.sh` | Shared paths and helpers of the jobs |
| `cluster/hpc` | Local helper: sync the code, submit and watch jobs (`cluster/hpc help`) |
| `third_party/` | Vendored `basicsr` and `ram` |

Tests: `python -m pytest` (only `tests/`; the archived tests are not collected).

## Cluster workflow

The cluster only runs committed code. `cluster/hpc sync` pushes the current branch and pulls it in `~/SegESR` on the
cluster; `cluster/hpc submit [-d] [VAR=val...] [sbatch options] <script>` syncs, submits and stores the commit in the
job comment. Logs go to `slurm_logs/<job name>_<jobid>.log`.
