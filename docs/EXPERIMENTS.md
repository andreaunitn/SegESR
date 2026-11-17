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
| 1052226 | e50a478 | segesr/checkpoint-15000 | full (refresh 750/500/250) | RealSR | 24.12 | 0.6885 | 0.3429 | 0.2557 | 156.77 | 6.12 | 0.6826 | 71.90 | 0.6045 | 1h13 for both datasets |
| 1052226 | e50a478 | segesr/checkpoint-15000 | full (refresh 750/500/250) | DRealSR | 26.71 | 0.7355 | 0.3706 | 0.2725 | 182.10 | 7.14 | 0.7129 | 69.14 | 0.5785 | |
| 1052234 | e50a478 | segesr/checkpoint-15000 | no SAM 2 refresh | RealSR | 24.16 | 0.6882 | 0.3433 | 0.2544 | 157.08 | 6.12 | 0.6828 | 71.79 | 0.6000 | |
| 1052234 | e50a478 | segesr/checkpoint-15000 | no SAM 2 refresh | DRealSR | 26.76 | 0.7352 | 0.3702 | 0.2702 | 179.66 | 7.04 | 0.7080 | 69.09 | 0.5726 | 59 min for both datasets |
| 1052235 | e50a478 | seesr_finetuned **final model (= step 30000)** | — | RealSR | 24.81 | 0.7118 | 0.3088 | 0.2261 | 129.70 | 5.44 | 0.6766 | 70.15 | 0.5602 | not checkpoint-15000: run_test.sh preferred the final model of the run folder over CHECKPOINTS (fixed afterwards) |
| 1052235 | e50a478 | seesr_finetuned **final model (= step 30000)** | — | DRealSR | 27.69 | 0.7592 | 0.3283 | 0.2387 | 157.03 | 6.56 | 0.6896 | 66.22 | 0.5261 | 34 min for both datasets |

**Findings (step 0.1):**
- At step 15000, SegESR loses fidelity against zero-shot SeeSR: −1.23 dB on RealSR, −1.62 dB on DRealSR, and LPIPS, DISTS and FID are worse. The no-reference metrics are better: MUSIQ, MANIQA and CLIPIQA.
- The SAM 2 refresh makes no measurable difference (24.12 against 24.16 dB on RealSR; 26.71 against 26.76 on DRealSR).
- The fine-tuned SeeSR final model also loses fidelity against zero-shot SeeSR (−0.54 / −0.64 dB). The comparison at the same step (checkpoint-15000) is still missing.

## Step V.0: VAE ceiling (SD 2 VAE, fp32)

| Job | Commit | Reconstruction | Dataset | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | FID↓ | NIQE↓ | CLIPIQA↑ | MUSIQ↑ | MANIQA↑ | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1052782 | 1ece733 | decoder(encoder(GT)) | RealSR | 35.39 | 0.9223 | 0.0365 | 0.0379 | 13.47 | 6.80 | 0.4680 | 57.39 | 0.3323 | 6 min for both datasets |
| 1052782 | 1ece733 | decoder(encoder(GT)) | DRealSR | 39.78 | 0.9496 | 0.0335 | 0.0351 | 12.30 | 7.49 | 0.4835 | 50.12 | 0.3147 | |
| 1052783 | 1ece733 | decoder(best latent), 200 steps | RealSR | | | | | | | | | | running, MSE about 2× lower than encode |
| 1052783 | 1ece733 | decoder(best latent), 200 steps | DRealSR | | | | | | | | | | queued in the same job |
