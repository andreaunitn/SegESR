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
| 1053093 | 343c6e6 | seesr_finetuned/checkpoint-15000 | — | RealSR | 24.82 | 0.7116 | 0.3092 | 0.2267 | 130.09 | 5.44 | 0.6806 | 70.16 | 0.5599 | 39 min for both datasets |
| 1053093 | 343c6e6 | seesr_finetuned/checkpoint-15000 | — | DRealSR | 27.71 | 0.7579 | 0.3282 | 0.2391 | 157.25 | 6.51 | 0.6918 | 66.39 | 0.5270 | |

**Findings (step 0.1):**
- At step 15000, SegESR loses fidelity against zero-shot SeeSR: −1.23 dB on RealSR, −1.62 dB on DRealSR, and LPIPS, DISTS and FID are worse. The no-reference metrics are better: MUSIQ, MANIQA and CLIPIQA.
- The SAM 2 refresh makes no measurable difference (24.12 against 24.16 dB on RealSR; 26.71 against 26.76 on DRealSR).
- The fine-tuned SeeSR final model also loses fidelity against zero-shot SeeSR (−0.54 / −0.64 dB).
- Fine-tuned SeeSR at checkpoint-15000 gives the same numbers as its final model (24.82 against 24.81 dB; 27.71 against 27.69).
- **At the same step, SegESR is 0.70 dB (RealSR) and 1.00 dB (DRealSR) below fine-tuned SeeSR**, with worse LPIPS, DISTS and FID. The segmentation conditioning does not help on these benchmarks.

## Step V.0: VAE ceiling (SD 2 VAE, fp32)

| Job | Commit | Reconstruction | Dataset | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | FID↓ | NIQE↓ | CLIPIQA↑ | MUSIQ↑ | MANIQA↑ | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1052782 | 1ece733 | decoder(encoder(GT)) | RealSR | 35.39 | 0.9223 | 0.0365 | 0.0379 | 13.47 | 6.80 | 0.4680 | 57.39 | 0.3323 | 6 min for both datasets |
| 1052782 | 1ece733 | decoder(encoder(GT)) | DRealSR | 39.78 | 0.9496 | 0.0335 | 0.0351 | 12.30 | 7.49 | 0.4835 | 50.12 | 0.3147 | |
| 1052783 | 1ece733 | decoder(best latent), 200 steps | RealSR | 39.23 | 0.9573 | 0.0520 | 0.0758 | 30.29 | 7.31 | 0.3932 | 57.54 | 0.3398 | 4h41 for both datasets |
| 1052783 | 1ece733 | decoder(best latent), 200 steps | DRealSR | 43.24 | 0.9714 | 0.0519 | 0.0764 | 31.21 | 7.96 | 0.4368 | 51.15 | 0.3307 | |

**Findings (V.0):**
- With the best latent, the SD 2 decoder reaches 39.23 dB on RealSR and 43.24 dB on DRealSR, 3.5–3.8 dB above decoder(encoder(GT)). LPIPS, DISTS and FID get worse, because the optimisation minimises only the MSE.
- The decoder can represent the GT far better than any SR model (24–28 dB): the error comes from the latent the diffusion model estimates, not from the VAE.
- Fine-structure metrics were not computed for this reconstruction: they cannot change the V.1 decision, which already fails with the weaker decoder(encoder(GT)).

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
- **[Superseded, see step H.0a]** The energy columns above are means of per-image ratios, which nearly flat images inflate. **The SR models put much more high-frequency energy than the GT.** In the `high` band they have 5.9–8.2× the GT's energy on RealSR and 7.4–12.3× on DRealSR, while the VAE stays slightly below 1. The excess grows with fine-tuning and with SegESR, in the same order as MUSIQ and MANIQA: invented detail or over-sharpening.

## Step H.0a: generated energy on a sharp GT (DIV2K)

Job 1053286 (commit 2e75e8a) ran `fine_structure.sh` on SeeSR's DIV2K outputs. Its per-image mean showed 8.95× in the `midlow` band, an artefact of nearly flat images. It was checked locally with robust statistics, on all 100 RealSR images and 100 DIV2K images (every 30th):

| Dataset | Band | Mean of per-image ratios | Median | Ratio of summed energies |
|---|---|---|---|---|
| RealSR | midhigh | 1.39 | 1.13 | 1.03 |
| RealSR | high | 5.92 | 5.02 | 4.48 |
| DIV2K | midhigh | 1.05 | 0.73 | 0.61 |
| DIV2K | high | 1.33 | 0.77 | 0.60 |

**Findings:**
- On RealSR the excess in the highest octave is real: 10th percentile 1.67, median 5.0.
- On DIV2K, whose GT is sharp, SeeSR generates *less* fine detail than the GT.
- So SeeSR is not over-generating in absolute terms. Its detail follows the sharp training images, and the camera-captured GTs of RealSR and DRealSR are softer.
- `fine_structure.py` now reports the ratio of summed energies (`energy_<band>`) and the median (`energy_<band>_median`).

**Re-aggregation (job 1053408, commit 86f9229).** The band PSNRs and edge metrics of V.1 are unchanged. Energy in the `high` band, as ratio of summed energies / median of per-image ratios:

| Model | RealSR | DRealSR | DIV2K |
|---|---|---|---|
| seesr | 4.48 / 5.02 | 7.08 / 4.93 | 0.63 / 0.85 |
| seesr_finetuned | 5.07 / 6.23 | 7.95 / 5.87 | |
| segesr_checkpoint-15000 | 5.51 / 6.09 | 9.96 / 9.25 | |
| segesr_checkpoint-15000_norefresh | 5.39 / 6.15 | 9.84 / 9.20 | |
| vae_sd2_encode | 0.85 / 0.91 | 0.91 / 0.96 | |

## Step H.0b: which metrics reward generated detail

Job 1053407, commit 86f9229: `cluster/band_scale.sh` with `DATASETS="RealSR DRealSR DIV2K"`, 8h18 (DIV2K: 3000 images per variant).

The frequencies above the LR Nyquist limit (1/8 cycle per HR pixel for ×4) of the SeeSR outputs are multiplied by alpha (`seesr_hf<alpha>`). The oracle scales each of the two octaves above the limit so that its energy equals the GT's, per image. `hf1` is the original output (same metrics: sanity check). RealSR and DRealSR have soft camera-captured GTs; DIV2K has sharp GTs.

`EDGE_F` and `ENERGY_HIGH` (ratio of summed energies) come from `finestruct_seesr_hf<alpha>.json`.

### RealSR (soft GT)
| Variant | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | FID↓ | NIQE↓ | CLIPIQA↑ | MUSIQ↑ | MANIQA↑ | EDGE_F↑ | ENERGY_HIGH |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hf0 | 26.88 | 0.7570 | 0.3351 | 0.2369 | 133.08 | 7.94 | 0.5014 | 37.79 | 0.2979 | 0.6028 | 0.27 |
| hf0.25 | **26.95** | **0.7647** | 0.2786 | 0.2052 | 118.16 | 7.12 | 0.3549 | 47.40 | 0.2677 | 0.6775 | 0.52 |
| hf0.5 | 26.66 | 0.7599 | 0.2678 | 0.2035 | **111.44** | 6.20 | 0.4223 | 60.53 | 0.3801 | 0.7219 | 1.30 |
| hf1 (SeeSR) | 25.35 | 0.7246 | 0.3001 | 0.2211 | 125.49 | 5.41 | 0.6644 | 69.49 | 0.5394 | 0.7475 | 4.48 |
| hf1.5 | 23.71 | 0.6689 | 0.3330 | 0.2335 | 129.46 | **4.86** | **0.7114** | 70.94 | 0.5728 | 0.7494 | 9.61 |
| hf2 | 22.26 | 0.6107 | 0.3637 | 0.2532 | 136.33 | 4.92 | 0.6636 | **71.20** | **0.5729** | 0.7343 | 15.62 |
| oracle | 26.17 | 0.7452 | **0.2307** | **0.1846** | 117.11 | 6.13 | 0.4781 | 63.73 | 0.3862 | **0.7654** | 1.10 |

### DRealSR (soft GT)
| Variant | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | FID↓ | NIQE↓ | CLIPIQA↑ | MUSIQ↑ | MANIQA↑ | EDGE_F↑ | ENERGY_HIGH |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hf0 | **29.95** | 0.8180 | 0.3151 | 0.2376 | 147.39 | 8.75 | 0.5148 | 36.58 | 0.2985 | 0.5263 | 0.71 |
| hf0.25 | 29.87 | **0.8186** | 0.2763 | 0.2112 | 136.23 | 8.15 | 0.4011 | 45.03 | 0.2802 | 0.5813 | 1.06 |
| hf0.5 | 29.52 | 0.8089 | 0.2740 | 0.2088 | **133.76** | 7.03 | 0.4648 | 56.09 | 0.3652 | 0.5921 | 2.25 |
| hf1 (SeeSR) | 28.33 | 0.7706 | 0.3164 | 0.2312 | 151.32 | 6.42 | 0.6743 | 65.24 | 0.5047 | 0.5869 | 7.08 |
| hf1.5 | 26.93 | 0.7193 | 0.3529 | 0.2462 | 157.28 | 5.91 | **0.7024** | 67.05 | 0.5354 | 0.5706 | 14.89 |
| hf2 | 25.62 | 0.6671 | 0.3836 | 0.2656 | 163.91 | **5.74** | 0.6559 | **67.61** | **0.5404** | 0.5472 | 24.13 |
| oracle | 29.18 | 0.7969 | **0.2509** | **0.1991** | 140.32 | 6.79 | 0.5029 | 59.20 | 0.3703 | **0.6090** | 1.43 |

### DIV2K (sharp GT)
| Variant | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | FID↓ | NIQE↓ | CLIPIQA↑ | MUSIQ↑ | MANIQA↑ | EDGE_F↑ | ENERGY_HIGH |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hf0 | **24.85** | 0.6373 | 0.4943 | 0.2715 | 43.07 | 7.98 | 0.5160 | 34.06 | 0.2737 | 0.4730 | 0.05 |
| hf0.25 | 24.83 | **0.6392** | 0.4050 | 0.2384 | 35.21 | 6.94 | 0.3761 | 43.51 | 0.2479 | 0.5491 | 0.09 |
| hf0.5 | 24.61 | 0.6317 | 0.3490 | 0.2131 | 28.36 | 5.86 | 0.4480 | 58.16 | 0.3459 | 0.6247 | 0.19 |
| hf1 (SeeSR) | 23.73 | 0.5993 | 0.3198 | 0.1968 | 26.16 | 4.80 | 0.6932 | 68.69 | 0.5041 | 0.6967 | 0.63 |
| hf1.5 | 22.58 | 0.5550 | 0.3296 | 0.2065 | 28.01 | **4.61** | **0.7212** | 70.34 | **0.5397** | 0.7222 | 1.34 |
| hf2 | 21.49 | 0.5105 | 0.3522 | 0.2219 | 31.51 | 4.78 | 0.6616 | **70.60** | 0.5358 | **0.7291** | 2.20 |
| oracle | 23.55 | 0.5893 | **0.3032** | **0.1917** | **26.12** | 4.81 | 0.6637 | 68.26 | 0.4915 | 0.7274 | 0.90 |

Oracle gains (mean / median over images):

| Dataset | [f_c, 2 f_c) | [2 f_c, Nyquist] |
|---|---|---|
| RealSR | 1.000 / 0.935 | 0.496 / 0.441 |
| DRealSR | 0.870 / 0.822 | 0.489 / 0.455 |
| DIV2K | 1.224 / 1.143 | 1.204 / 1.052 |

**Findings (H.0b):**
- **Soft GTs reward removing generated detail.** On RealSR and DRealSR, PSNR and SSIM peak at hf0–hf0.25 (+1.6 dB). LPIPS, DISTS and FID also prefer hf0.5 or the oracle to the original output. The oracle improves fidelity and perception together: +0.82 dB and LPIPS 0.300 → 0.231 on RealSR; +0.85 dB and 0.316 → 0.251 on DRealSR.
- **On the sharp GT the gain disappears.** On DIV2K the oracle slightly amplifies the detail (median gains 1.05–1.14) and gives only LPIPS −0.017 and DISTS −0.005, at −0.18 dB. Removing detail costs a lot of perception there (hf0: LPIPS 0.320 → 0.494, FID 26 → 43).
- So most of the oracle's gain on RealSR and DRealSR comes from matching the softness of the camera-captured GTs, not from generating better detail. Part of the perception-distortion trade-off measured on these benchmarks is this mismatch.
- **No-reference metrics reward more generated detail on every dataset:** NIQE, CLIPIQA, MUSIQ and MANIQA keep improving up to hf1.5–hf2, whatever the GT's amount of detail.
- PSNR is always best with little or no generated detail, also on DIV2K: the classic perception-distortion trade-off.
- **Decision:** a per-image calibration of the generated detail (H.1) is not worth a method. Even with the GT, the gain on sharp GTs is small; a predictor from the LR would recover less.
