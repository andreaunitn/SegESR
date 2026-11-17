# Experiment log

One row per cluster job and dataset, newest last. The commit is the one the job ran (`cluster/hpc jobs` shows it as `code:<rev>`).
Metrics come from `metrics/<dataset>/results_<name>.json` (`data_tools/metrics.py`: PSNR and SSIM on the Y channel with a 4-pixel crop, then pyiqa).
The step ids refer to [ROADMAP.md](ROADMAP.md).

↑ higher is better, ↓ lower is better.

## Reference: zero-shot SeeSR

| Job | Commit | Model | Dataset | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | FID↓ | NIQE↓ | CLIPIQA↑ | MUSIQ↑ | MANIQA↑ | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | — | preset/models/seesr | RealSR | 25.35 | 0.7246 | 0.3001 | 0.2211 | 125.49 | 5.41 | 0.6644 | 69.49 | 0.5394 | computed 2026-10-08 |
| — | — | preset/models/seesr | DRealSR | 28.33 | 0.7706 | 0.3164 | 0.2312 | 151.32 | 6.42 | 0.6743 | 65.24 | 0.5047 | computed 2026-10-08 |

SRSR (NeurIPS'25) reports zero-shot SeeSR at:
- 25.18 dB / LPIPS 0.3009 on RealSR;
- 28.17 dB / LPIPS 0.3189 on DRealSR.

Our numbers are about 0.16 dB higher on both datasets. Step 0.3 has to explain the difference (crop, Y channel, color fix, seed).

SRSR-SeeSR reaches 26.40 / 0.2718 on RealSR and 29.50 / 0.2866 on DRealSR.

## Step 0.1: baseline at `checkpoint-15000`

All jobs use `configs/test_default.yaml`: 50 steps, guidance 5.5, LR start, seed 42.

| Job | Commit | Model | Variant | Dataset | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | FID↓ | NIQE↓ | CLIPIQA↑ | MUSIQ↑ | MANIQA↑ | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1052226 | e50a478 | segesr/checkpoint-15000 | full (refresh 750/500/250) | RealSR | | | | | | | | | | queued |
| 1052226 | e50a478 | segesr/checkpoint-15000 | full (refresh 750/500/250) | DRealSR | | | | | | | | | | queued |
| 1052234 | e50a478 | segesr/checkpoint-15000 | no SAM 2 refresh | RealSR | | | | | | | | | | queued |
| 1052234 | e50a478 | segesr/checkpoint-15000 | no SAM 2 refresh | DRealSR | | | | | | | | | | queued |
| 1052235 | e50a478 | seesr_finetuned/checkpoint-15000 | — | RealSR | | | | | | | | | | queued |
| 1052235 | e50a478 | seesr_finetuned/checkpoint-15000 | — | DRealSR | | | | | | | | | | queued |
