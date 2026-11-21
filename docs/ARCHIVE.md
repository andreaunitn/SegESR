# Archive of the closed directions

Every direction that was tried and closed keeps its code in `archive/<direction>/` and its models, outputs, metrics
and logs on the cluster in `preset/archive/<direction>/`. Results and decisions are in [ROADMAP.md](ROADMAP.md) and
[EXPERIMENTS.md](EXPERIMENTS.md).

**Rerunning archived code.** Inside `archive/` each file keeps its path relative to the repository root
(`archive/track_a/data_tools/seed_spread.py` was `data_tools/seed_spread.py`), but the archived code is not meant
to run from there. The git tag `pre-archive` is the last commit with everything at its original paths:
`git worktree add ../SegESR-pre-archive pre-archive` gives a copy where every archived step runs as recorded.
Track M and Track R were archived later: their tags are `pre-archive-m` and `pre-archive-r`.

**Reading archived metrics.** `python3 data_tools/results_table.py --metrics_dir preset/archive/<direction>/metrics`
(add `--prefix finestruct` or `--prefix seedspread` for the other files).

| Direction | Status | Steps | Code | Data on the cluster |
|---|---|---|---|---|
| Segmentation (SegESR) | on hold since step 0.1 | 0.1 | `archive/segmentation/` | `preset/archive/segmentation/` |
| Track V (VAE) | paused after V.1 | V.0, V.1 | `archive/track_v/` | `preset/archive/track_v/` |
| Track H (hallucinations) | stopped after H.0b | H.0a, H.0b | `archive/track_h/` | `preset/archive/track_h/` |
| Track A (LR-supported band), with T.0 | method dropped | A.0, A.0b, T.0 | `archive/track_a/` | `preset/archive/track_a/` |
| BiBandSR | stopped at gate O.1 | O.0, O.1 | `archive/bibandsr/` | `preset/archive/bibandsr/` |
| Track M (where to generate) | stopped at gate M.0b | M.0, M.0b | `archive/track_m/` | `preset/archive/track_m/` |
| Track R (representation alignment) | stopped at gate R.0 | R.0 | `archive/track_r/` | `preset/archive/track_r/` |

Each data folder has the same layout: `train_output/` (checkpoints), `output/<name>/<dataset>/` (SR images, as in
`preset/datasets/output/`), `metrics/<dataset>/` (as in `metrics/`), `models/`, `tensorboard/`, `slurm_logs/`.

## What stays in place

These are the reference points of any new direction, so they stay where the active scripts look for them:

- outputs and metrics of zero-shot SeeSR (`seesr`), PiSA-SR (`pisasr`), OSEDiff (`osediff`), FreqOrtho-SR at its
  default setting (`freqortho`), FiDeSR with its official empty prompt (`fidesr`), bicubic (`bicubic`) and the four O.1 checkpoints (`o1_pisasr`, `o1_split`, `o1_pisasr_500`, `o1_split_500`);
- the base models in `preset/models/` (SD 2 base, SD 2.1 base, SeeSR, DAPE, RAM, PiSA-SR, OSEDiff, FreqOrtho-SR, FiDeSR);
- the external code in `preset/external/` (PiSA-SR, OSEDiff, FreqOrtho-SR, FiDeSR) and every dataset in `preset/datasets/`;
- the logs of setup, downloads, zero-shot SeeSR and O.0.

## Segmentation (SegESR)

SegESR (SeeSR with SAM 2 conditions, PAFB blocks and a SAM 2 perceptual loss) and the fine-tuned SeeSR baseline.

- Code: the `segesr` package, `train.py`, `test.py` (also the generator of every SeeSR output of the other
  directions), `configs/`, `scripts/`, `cluster/{train,test,prepare_data,tensorboard}.sh`, the data preparation in
  `data_tools/` (`make_paired_data.py`, `make_tags.py`, `make_dape_embeds.py`, `sam_processing.py`), their tests,
  the original `README.md` with its figure, and `commands.txt`.
- Data: `train_output/{segesr,seesr_finetuned}`; outputs and metrics of `segesr_checkpoint-15000`,
  `segesr_checkpoint-15000_norefresh`, `seesr_finetuned`, `seesr_finetuned_checkpoint-15000`; `models/tiny_vae`
  (used only by the SegESR losses); `tensorboard/`; the training, test and data logs.
- `data_tools/metrics.py --tb_dir` still imports its image helpers from `archive/segmentation/segesr`.

## Track V (VAE)

- Code: `data_tools/vae_ceiling.py`, `cluster/vae_ceiling.sh`, its test.
- Data: outputs and metrics of `vae_sd2_encode` and `vae_sd2_optimize`; the logs of V.0 and V.1.

## Track H (hallucinations)

- Code: `data_tools/band_scale.py` (also used by Track A, T.0 and BiBandSR), `cluster/band_scale.sh`, its test.
- Data: outputs and metrics of `seesr_hf0`, `seesr_hf0.25`, `seesr_hf0.5`, `seesr_hf1`, `seesr_hf1.5`,
  `seesr_hf2`, `seesr_hforacle`; the logs of H.0a and H.0b.

## Track A (LR-supported band), with T.0

- Code: `data_tools/{seed_spread,bicubic,regression_sr,rrdbnet}.py`,
  `cluster/{low_band,low_band_proxy,regression_fusion}.sh`, their tests, and `a0b_combo.sh` (the A.0b combination,
  run from the cluster home; only in the data folder).
- Data: outputs and metrics of `seesr_g1`, `seesr_g3`, `seesr_s3`, `seesr_nofix`, `seesr_wavelet`,
  `seesr_lowgt8`, `seesr_lowgt16`, `seesr_s3mean`, `seesr_s3lowmean8` (A.0); `seesr_g1low8`, `seesr_g1low16`,
  `seesr_bic16`, `seesr_g1low8_bic16` (A.0b); `realesrnet`, `bsrnet`, `seesr_realesrnetlow8`,
  `seesr_realesrnetlow16`, `seesr_bsrnetlow8`, `seesr_bsrnetlow16` (T.0); `seedspread_seesr_s3.json`;
  `models/regression` (RealESRNet, BSRNet); the logs.

## BiBandSR

- Code: the `bibandsr` package (`train.py`, `bands.py`), `cluster/bibandsr_train.sh`, their tests.
  `cluster/onestep_baselines.sh` and `cluster/setup_bibandsr.sh` stay active: they serve PiSA-SR and OSEDiff.
- Data: `train_output/bibandsr/` (`o1_pisasr`, `o1_split`, `o1_smoke_split`: checkpoints every 250 steps,
  `val.jsonl`, `calibration.json`, `log.jsonl`, `config.json`); the O.1 logs. The outputs and metrics of the O.1
  checkpoints stay in place.

## Track M (where to generate)

- Code: `data_tools/oracle_map.py`, `cluster/oracle_map.sh`, the tests of the oracle. `data_tools/curve_gap.py` and its
  tests (`tests/test_curve_gap.py`) stay active: they compare any model with a curve of settings at the same
  perceptual score. FreqOrtho-SR stays installed (`cluster/setup_bibandsr.sh`, `cluster/onestep_baselines.sh` with
  `LAMBDA_SEM`).
- Data: outputs and metrics of `freqortho_ls0`, `freqortho_ls0.5`, `freqortho_ls1`, `freqortho_ls1.5` (M.0 sweep),
  `freqortho_oracle64`, `freqortho_shuffle64`, `freqortho_oracle128` (M.0), `freqortho_budget64_b0.75`,
  `freqortho_budget64_b1`, `freqortho_budget64_b1.25` (M.0b), with the raw outputs of the sweep in `external_raw/`;
  the logs of M.0 and M.0b. The default `freqortho` stays in place.

## Track R (representation alignment)

- Code: `data_tools/feature_alignment.py`, `cluster/feature_alignment.sh`, their test. `data_tools/fidesr_test.py`
  stays active: `cluster/onestep_baselines.sh` runs FiDeSR through it (`fidesr`, and the prompts `fidesr_ramlr` and
  `fidesr_ramhr`). `cluster/setup_trackr.sh` became `cluster/setup_fidesr.sh`, without DINOv2.
- Data: outputs and metrics of `fidesr_ramlr` and `fidesr_ramhr` (R.0(a)) and `metrics/DIV2K/alignment_fidesr/`
  (R.0(b): `alignment.json`, `per_image.csv`); `models/dinov2-base`; the logs of R.0. The default `fidesr` stays in
  place.
