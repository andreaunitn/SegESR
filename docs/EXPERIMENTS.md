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

## Step V.1: fine-structure metrics (bands and edges)

Job 1053173, commit 2e75e8a: `cluster/fine_structure.sh`, CPU only, 1 min 37 s. Y channel, 4-pixel crop.

Bands are octaves of the spatial frequency, from `low` [0, 1/16) to `high` [1/4, 1/2] cycles per pixel. `energy_<band>` is the SR spectral energy divided by the GT's.

### DRealSR
| Model | PSNR_LOW↑ | PSNR_MIDLOW↑ | PSNR_MIDHIGH↑ | PSNR_HIGH↑ | EDGE_F↑ | EDGE_PSNR↑ | ENERGY_LOW | ENERGY_MIDLOW | ENERGY_MIDHIGH | ENERGY_HIGH |
|---|---|---|---|---|---|---|---|---|---|---|
| seesr | 34.85 | 33.95 | 33.58 | 37.36 | 0.5869 | 21.81 | 0.994 | 1.206 | 1.910 | 7.431 |
| seesr_finetuned | 34.11 | 33.27 | 32.98 | 36.66 | 0.5750 | 21.06 | 0.993 | 1.363 | 2.260 | 8.606 |
| segesr_checkpoint-15000 | 33.12 | 32.29 | 32.30 | 34.97 | 0.5479 | 20.40 | 0.991 | 1.710 | 2.664 | 12.253 |
| segesr_checkpoint-15000_norefresh | 33.22 | 32.32 | 32.32 | 35.05 | 0.5442 | 20.47 | 0.991 | 1.672 | 2.644 | 11.998 |
| vae_sd2_encode | **49.71** | **46.99** | **43.07** | **47.25** | **0.8658** | **34.92** | 0.997 | 0.931 | 0.870 | 0.931 |

### RealSR
| Model | PSNR_LOW↑ | PSNR_MIDLOW↑ | PSNR_MIDHIGH↑ | PSNR_HIGH↑ | EDGE_F↑ | EDGE_PSNR↑ | ENERGY_LOW | ENERGY_MIDLOW | ENERGY_MIDHIGH | ENERGY_HIGH |
|---|---|---|---|---|---|---|---|---|---|---|
| seesr | 33.00 | 31.81 | 29.82 | 32.86 | 0.7475 | 20.89 | 0.993 | 1.004 | 1.386 | 5.918 |
| seesr_finetuned | 32.36 | 31.19 | 29.33 | 32.21 | 0.7447 | 20.28 | 0.991 | 1.083 | 1.624 | 6.853 |
| segesr_checkpoint-15000 | 31.74 | 30.16 | 28.86 | 31.43 | 0.7216 | 19.68 | 0.989 | 1.262 | 1.713 | 8.173 |
| segesr_checkpoint-15000_norefresh | 31.80 | 30.18 | 28.87 | 31.52 | 0.7192 | 19.74 | 0.989 | 1.257 | 1.700 | 7.895 |
| vae_sd2_encode | **47.72** | **42.48** | **38.46** | **41.85** | **0.9252** | **31.29** | 0.997 | 0.934 | 0.880 | 0.888 |

**Findings:**
- **Gate V.1 (frequencies and edges) fails.** The VAE reconstruction of the GT holds only about 10–13% of SeeSR's error in the `high` band (RealSR 12.6%, DRealSR 10%) and about 9% around edges, against the 30% threshold. The OCR part of V.1 is still to do.
- **Where the error is.** For every SR model the largest error is in the `midhigh` band, not in `high`.
- **The SR models put much more high-frequency energy than the GT.** In the `high` band they have 5.9–8.2× the GT's energy on RealSR and 7.4–12.3× on DRealSR, while the VAE stays slightly below 1. The excess grows with fine-tuning and with SegESR, in the same order as MUSIQ and MANIQA: invented detail or over-sharpening.
