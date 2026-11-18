# SegESR roadmap

Five research directions share Phase 0.

- **Active: BiBandSR.** A one-step SR model trained with losses split at the LR Nyquist limit: fidelity below it, realism above it. See [BiBandSR](#bibandsr-one-step-sr-with-losses-split-at-the-lr-nyquist-limit-active).
- **Track A (fidelity in the LR-supported band): analysis done, training-free method dropped.** About half of SeeSR's MSE is at frequencies the LR determines, but changes at inference recover only +0.3–0.5 dB. T.0 shows that a model trained for fidelity does not reach that band of the RealSR and DRealSR GTs either; open: the same check on DIV2K (synthetic degradations, sharp GT), or a new direction. See [Track A](#track-a-fidelity-in-the-lr-supported-band-analysis-done).
- **On hold: the segmentation track (Phases 1–5 below).** At the same step, SegESR is below fine-tuned SeeSR (step 0.1).
- **Paused: Track V (VAE).** A decoder that can be plugged into any latent SR model, where each pixel uses as much LR evidence as the LR actually holds. Gate V.1 failed; V.2 was not tried, and PGSR already covers it. See [Track V](#track-v-evidence-aware-decoder-paused).
- **Stopped: fallback track H.** H.0b shows that the excess of generated detail is mostly a mismatch with the soft GTs of RealSR and DRealSR, so H.1 is dropped. Its analysis of the metrics stays available as supporting material.

## How we work

**One step = one question answered or one capability added**, small enough to finish in 1–2 days.
Each step has a card:

| Field | Content |
|---|---|
| Goal | What the step adds or which question it answers |
| Success criterion | A measurable outcome decided *before* running it |
| Changes | Files touched (few) |
| Tests | pytest checks that must pass |
| Experiment | Job(s) to run, if a GPU is needed (logged in [EXPERIMENTS.md](EXPERIMENTS.md)) |
| Result and decision | Numbers, then: go on, fix, or drop |

**Rules**

1. **Every new feature sits behind a flag that is off by default.** Old checkpoints and results stay reproducible, and every flag is an ablation for the paper.
2. **One branch per step**, `step/<id>-<name>` (e.g. `step/2.2-reliability-routing`). It is merged into `main` only after review. A failed step is a deleted branch.
3. **Every job goes into [EXPERIMENTS.md](EXPERIMENTS.md)** with its job id, commit, flags and metrics. `cluster/hpc submit` already stores the commit in the job comment.
4. **Gates.** At the end of each phase we decide, from the numbers, whether to continue as planned.

**Routine for a step:** write the code, test it, review the diff and commit message, push, run the job (only after an explicit OK), then record the results and update this file.

Status: `todo` · `doing` · `done` · `dropped`

## Phase 0: solid baseline

| Step | Goal | Status |
|---|---|---|
| 0.1 | Test SegESR (with and without SAM 2 refresh) and fine-tuned SeeSR at `checkpoint-15000` on RealSR + DRealSR | done |
| 0.2 | Script that collects all `metrics/*/results_*.json` into one table (`data_tools/results_table.py`) | done |
| 0.3 | Align with the SRSR protocol: zero-shot SeeSR must give 25.18 dB on RealSR | todo |
| 0.4 | Runtime per image in `test.py` (only measurable during generation), plus TOPIQ and LIQE in `metrics.py` (already in pyiqa 0.1.13). A-FINE needs a newer pyiqa: added before the final comparisons, recomputing every metric with the same version | todo |

**Gate 0:**
- Does SegESR clearly beat fine-tuned SeeSR at the same number of steps? **No**: −0.70 dB on RealSR, −1.00 dB on DRealSR, worse LPIPS, DISTS and FID.
- Does the SAM 2 refresh help? **No** measurable difference.
- Where does the README's +2.22 dB come from (dataset, protocol)? Still open (step 0.3): it does not hold on RealSR and DRealSR.

## BiBandSR: one-step SR with losses split at the LR Nyquist limit (active)

For ×4 SR the LR holds the spatial frequencies below 1/8 cycle per HR pixel (its Nyquist limit); above it, the content is invented. Our measurements on SeeSR support splitting the losses there:
- above the limit, 73–82% of the error is variance between seeds, so a pixel loss there only pushes towards the average (blur);
- about half of the MSE is below the limit (Track A);
- classifier-free guidance moves the band below the limit (A.0), and distillation losses such as CSD and VSD use it (CFG 7.5).

The model: fidelity losses only below the limit, realism losses (LPIPS, CSD) only above it. In the SD latent the 1/8 cutoff cannot be represented directly (the latent only reaches 1/16 of the HR pixels), so the realism gradient is projected in pixel space through the decoder.

Base: **PiSA-SR** (CVPR'25, Apache-2.0, SD 2.1-base). It already separates fidelity and semantics, but in parameters (a pixel LoRA trained with L2, a semantic LoRA with LPIPS + CSD, both on all frequencies). Its CSD loss needs one frozen UNet only, lighter than the VSD of OSEDiff. **OSEDiff** (Apache-2.0) is a second baseline, inference only.

Environment: `~/venvs/bibandsr` (`requirements-bibandsr.txt`, `cluster/setup_bibandsr.sh`), on the same container: PiSA-SR and OSEDiff need diffusers 0.25 and peft 0.9. Downloaded code and weights go to `preset/` (`preset/external/`, `preset/models/`).

| Step | Goal | Status |
|---|---|---|
| O.0 | PiSA-SR and OSEDiff with their official weights, scored with our protocol on RealSR, DRealSR and DIV2K: metrics, band errors and energy (`data_tools/collect_sr.py`, `cluster/onestep_baselines.sh`) | doing |
| O.1 | Two identical short fine-tunings from the official PiSA-SR weights, on our data and hardware: the original losses (control) and the split losses. Checkpoints and automatic resubmission past the 1-day limit | todo |
| O.2 | If O.1 shows a gain: the split losses behind flags in a full training; ablations (fidelity split only, realism split only, cutoff 1/16 or 1/8) | todo |
| O.3 | The same on OSEDiff; comparisons (PiSA-SR, OSEDiff, SeeSR, others); writing | todo |

**Gate O.0.** Go on if, in the one-step models too, the band below 1/8 holds a large part of the error (at least about 30% of the MSE), as in SeeSR.

## Track A: fidelity in the LR-supported band (analysis done)

For ×4 SR, the frequencies below 1/8 cycle per HR pixel are determined by the LR. The band PSNRs of V.1 put about 43% (RealSR), 54% (DRealSR) and 41% (DIV2K) of SeeSR's MSE there, and 15–24% in the lowest octave [0, 1/16) alone. The VAE reconstructs these bands almost exactly (47–50 dB in the lowest one), so the error comes from sampling. SegESR and fine-tuned SeeSR lose fidelity mostly there too.

Goal: find what moves the LR-supported band during sampling and reduce it without training, for any model, without knowing the degradation. Related: FiDeSR (CVPR'26: low-frequency consistency in one-step models, with training); ILVR and DDNM (low band from the LR, with a known degradation); the wavelet color fix of StableSR.

| Step | Goal | Status |
|---|---|---|
| A.0 | Diagnosis on zero-shot SeeSR (RealSR + DRealSR): guidance 1 / 3 / 5.5; color fix adain / wavelet / none; 3 seeds, split of each band error into a seed part and a systematic part (`data_tools/seed_spread.py`); oracle with the band below 1/16 or 1/8 from the GT and a fusion with the mean of the seeds (`band_scale.py --low_from`, `cluster/low_band.sh`). **Gate passes with a small margin**: guidance 1 gives +0.67 / +0.69 dB below 1/8, the mean of 3 seeds +0.61 / +0.78; about 80% of the error is systematic | done |
| A.0b | Before changing the sampler: the band below 1/16 or 1/8 of guidance 1 combined with the band above of guidance 5.5 (what a frequency-split guidance could give), and the band errors of the bicubic LR (`data_tools/bicubic.py`, `cluster/low_band_proxy.sh`). Go on to A.1 if the combination gains at least about 0.6 dB of PSNR and keeps MUSIQ and MANIQA within about 1 point of SeeSR. **Fails**: +0.26 / +0.36 dB; with the lowest octave from the bicubic LR +0.48 / +0.50 dB, but LPIPS, MANIQA and FID get worse | done |
| A.1 | Training-free method (frequency-split guidance, low band from the seeds or the LR). **Dropped**: at most about +0.3–0.5 dB (A.0b) | dropped |
| T.0 | Feasibility of a trained reconstruction, without training: the band below 1/16 or 1/8 of SeeSR taken from regression SR models trained with a pixel loss on realistic degradations (RealESRNet_x4plus, BSRNet; `data_tools/regression_sr.py`, `data_tools/rrdbnet.py`, `cluster/regression_fusion.sh`). **Fails on RealSR and DRealSR**: at most +0.28 / +0.09 dB, with worse FID and MANIQA | done |
| T.1a | If T.0 passes: a small restorer trained only on the band below 1/8 (frequency-limited loss), fused with the output of any diffusion model; it cannot change the generated detail above the LR Nyquist limit | todo |
| T.1b | If T.0 passes: a fidelity loss limited to the band below 1/8 inside the training of SeeSR (on the decoded x0 prediction, on crops to fit 24 GB) | todo |

**Gate T.0.** Go on to T.1 if the fusion with the regression low band gains at least **+1.0 dB** of PSNR on both RealSR and DRealSR, with LPIPS no worse and MUSIQ and MANIQA within about 1 point of SeeSR. Otherwise a model trained for fidelity does not reconstruct that band better than SeeSR, and the oracle headroom is out of reach from the LR: rethink the direction.

**Gate A.0.** Continue if a realistic setting (no GT) gains at least **+0.5 dB in the band below 1/8** on both RealSR and DRealSR, with LPIPS no worse than +0.005. A.1 depends on the cause:
- **Guidance:** classifier-free guidance applied only above the LR Nyquist limit.
- **Seed:** a seed part of the low-band error of at least about 30% calls for a consistency step on the low band during sampling, without a known degradation.
- **Neither, and the oracle gain is large:** the degradation blur, i.e. degradation estimation. Stop: crowded field.

## Segmentation track (on hold)

Method: **B + C + D**. The evaluation contribution is **A**.

- **B** gives every SAM 2 segment a reliability score.
- **C** controls fidelity versus generation per segment, driven by that score.
- **D** generates in two steps (draft, SAM 2 re-check on the draft, refinement).
- **A** is a segment-level hallucination metric.

The novelty is the per-segment reliability and its use to decide how much to generate in each segment. The closed loop and the two steps are the setting, not the contribution: InstantIR and SegSR already re-estimate conditions during sampling. See the survey notes for the related work: SRSR (NeurIPS'25), SAMSR (IJCAI'25), HoliSDiP, SegSR, InstantIR, MMSR (CVPR'25), CODSR, DTPSR (CVPR'26) and RASS (IJCV'26).

## Phase 1: metric A (segment-level hallucinations)

| Step | Goal | Status |
|---|---|---|
| 1.1 | Independent segmenter (Mask2Former or OneFormer, not SAM 2) run on SR outputs and GT | todo |
| 1.2 | Segment matching between SR and GT: spurious and missing segments, boundary F-score, per-segment feature distance | todo |
| 1.3 | Metric in `data_tools/metrics.py` behind a flag | todo |
| 1.4 | Sanity checks: GT against itself is perfect; blur is not counted as hallucination; a pasted object is detected | todo |

## Phase 2: B (segment reliability)

| Step | Goal | Status |
|---|---|---|
| 2.1 | Reliability score r per segment from the SAM 2 predicted IoU and stability score (kept in `compute_sam2_conditions`) | todo |
| 2.2 | Add `log(r)` to the routing bias in `segment_routed_attention`, behind a flag | todo |
| 2.3 | Inference-only experiment on `checkpoint-15000` (no new training) | todo |
| 2.4 | LR-to-draft consistency of each segment inside the existing SAM 2 refresh, used to update r | todo |
| 2.5 | Degradation-robust Hiera: LoRA trained so that Hiera(LR) ≈ Hiera(GT); targets are in `sam_embeds_gt/` | todo |

**Gate 2:** does reliability-weighted routing help, with and without the robust Hiera?

## Phase 3: D (two-step generation)

| Step | Goal | Status |
|---|---|---|
| 3.1 | Run the current model with 2 timesteps (no training) to measure the gap | todo |
| 3.2 | Train step 1 as a one-step generator from the LR | todo |
| 3.3 | Train step 2 on the real draft of step 1, with SAM 2 conditions from the draft (replaces `clean_sam_prob`) | todo |
| 3.4 | CSD distillation (PiSA-SR style, one frozen UNet, fits 24 GB) | todo |

**Gate 3:** quality of 2 steps against 50 steps, and runtime.

## Phase 4: C (per-segment fidelity control)

| Step | Goal | Status |
|---|---|---|
| 4.1 | Extra input channel with the control map c in the ControlNet (zero-initialised, behind a flag), plus a weight-transfer test | todo |
| 4.2 | Per-segment loss weighting: (1 - c) · pixel loss + c · perceptual loss | todo |
| 4.3 | c computed automatically from the reliability scores of B | todo |

**Gate 4:**
- Does reliability-driven c reduce hallucinations (metric A)?
- Does it beat these alternative weights:
  - the SAMSR mask-coverage weights;
  - the binary SRSR grounding;
  - a single global score, as in InstantIR?

## Phase 5: baselines, ablations, paper

| Step | Goal | Status |
|---|---|---|
| 5.1 | Baselines: SeeSR, PiSA-SR, OSEDiff, FaithDiff, DiT4SR, TVT, SRSR, SAMSR, HoliSDiP, InstantIR, MMSR/CODSR/DTPSR if code exists | todo |
| 5.2 | Ablations (see below) | todo |
| 5.3 | Metrics: PSNR, SSIM, LPIPS, DISTS, A-FINE, CLIPIQA, MUSIQ, MANIQA, metric A, seed variance, runtime | todo |
| 5.4 | Writing | todo |

Ablations for 5.2:
- SICA against the NTIRE'25 approach;
- SAM 2 against DINOv2 against CLIP embeddings;
- SAM masks against SLIC superpixels;
- SAM loss against CLIP and DINO losses;
- reliability weights against SAMSR, SRSR and global (InstantIR) weights.

## Track V: evidence-aware decoder (paused)

The SD VAE compresses by 8× and loses fine structures (small text, thin lines, textures). No diffusion prior can recover them (TVT, ICCV'25). What already exists:

| Approach | Works |
|---|---|
| Less compression | TVT (4× VAE) |
| More latent channels | DPIR, TOC-SR, LCUDiff (16-channel VAEs) |
| Degradation-robust encoder | OSEDiff, DPIR, SUPIR |
| LR features into the decoder | StableSR CFW (one global weight), Diff-Restorer, GuideSR (full-resolution branch) |
| Pre-VAE pixel evidence | PGSR (NeurIPS'26, FLUX): injected uniformly, no gating, no inference control, not plug-in |

**Open, and our target:**
- **A decoder that decides per pixel how much to trust the LR**, from how informative the LR is there, with inference-time control.
- **Plugging it into several frozen latent SR models** (SeeSR, SegESR, OSEDiff, PiSA-SR) without retraining them.
- **A dataset-level analysis** of how much of the SR error comes from the VAE.

| Step | Goal | Status |
|---|---|---|
| V.0 | VAE ceiling on RealSR and DRealSR: decoder(encoder(GT)) and decoder(best latent), scored like SR outputs (`data_tools/vae_ceiling.py`, `cluster/vae_ceiling.sh`). Compare with the SeeSR and SegESR outputs. **The best latent reaches 39.2 / 43.2 dB** | done |
| V.1 | Fine-structure benchmark: error per frequency band, edge F-score, OCR on small text (as PGSR, TextOCR). Bands and edges: `data_tools/fine_structure.py`, `cluster/fine_structure.sh`. **Bands and edges fail the gate** (VAE share of the error 9–13%); OCR postponed | done |
| V.2 | Baseline decoder adapter: LR features injected uniformly into the frozen SD VAE decoder (CFW- and PGSR-like), trained on latents of a frozen SR model | todo |
| V.3 | Evidence-aware gating: a per-pixel gate predicted from the LR (how informative it is locally), with a global scale for inference-time control | todo |
| V.4 | Plug-in test: the same decoder on other frozen models (SeeSR, SegESR, OSEDiff, PiSA-SR), with and without light fine-tuning | todo |
| V.5 | Baselines (StableSR CFW, GuideSR, TVT, PGSR on SD 2 if its code allows), ablations, writing | todo |

**Gate V.0 (done).**
- The VAE reconstruction of the GT reaches 35.39 dB / LPIPS 0.0365 on RealSR and 39.78 dB / 0.0335 on DRealSR.
- With the best latent the decoder reaches 39.23 dB on RealSR and 43.24 dB on DRealSR.
- Every SR model is at 24–28 dB.
- On whole images, the VAE explains only a small part of the error (about 7–10% of the MSE). Most of it comes from the latent estimated by the diffusion model.
- So "the VAE loses details" is a weak motivation on average. The track continues only if one of the two gates below passes.

**Gate V.1: does the VAE matter on fine structures?** Go on if at least one of these holds on RealSR + DRealSR:
- **Small text:** the OCR word accuracy of decoder(encoder(GT)) is at least 10 points below the GT's.
- **High frequencies:** the VAE reconstruction already holds at least 30% of the SR models' error in the highest frequency band (or in thin-edge regions).

**Gate V.2: can the decoder correct the latent with the LR?** A decoder adapter with uniform LR injection, on frozen SeeSR, must give:
- at least **+0.5 dB PSNR** on both RealSR and DRealSR;
- LPIPS no worse than +0.005.

It must get there within about one week of work. PGSR reports +2.40 dB on an SD 2.1 model, so a smaller gain means our setup does not reproduce the effect.

**Decision:**
- **V.1 passes:** the motivation is the VAE losing fine structures. Continue with V.2 and the following steps.
- **V.1 fails, V.2 passes:** the motivation becomes the decoder as the place where the LR corrects errors of the diffusion model, not VAE capacity. Continue with V.3–V.5.
- **Both fail:** stop Track V and switch to the fallback below.

## Fallback track H: where latent SR models fail (error decomposition and hallucinations)

Goal: an analysis, plus a model-agnostic tool. It splits the error of latent SR models into three parts:
- what the VAE loses (V.0 already measures it);
- the error in the estimated latent;
- invented content (hallucinations).

It also locates hallucinations per pixel, for example from disagreement between samples or from consistency with the LR after re-degradation, and corrects them where the LR contradicts the output. It runs on frozen models (SeeSR, OSEDiff, PiSA-SR, SegESR…).

It reuses everything done so far:
- `vae_ceiling.py` and the SR outputs;
- the metrics pipeline;
- metric A of the segmentation track (segment-level hallucinations).

It needs little GPU time. The step 0.1 results point there: fine-tuning moves SeeSR and SegESR towards perception (MUSIQ and MANIQA up, PSNR, LPIPS and DISTS worse).

Before starting it: a targeted literature check on uncertainty and conformal prediction for restoration, and on HalluGen (CVPR'26).

**Literature check (done).** Hallucination maps and scores exist:
- Conformal SR (NeurIPS'25);
- Hallucination Score (MLLM-based, preprint);
- CHEM;
- HalluGen (CVPR'26, MRI).

Nobody measures the **excess of generated high-frequency energy** of diffusion SR models against the GT, or checks which metrics reward it.

**Key fact.** For ×4 SR the LR holds frequencies only below 1/8 cycle per HR pixel, so everything above it is generated. On RealSR and DRealSR the models generate 4.5–10× the GT's energy in the highest octave, more after fine-tuning and with SegESR. On DIV2K, whose GT is sharp, SeeSR generates 0.6× (H.0a).

| Step | Goal | Status |
|---|---|---|
| H.0a | V.1 metrics on DIV2K (sharp GT): does the excess of generated energy hold, or is it due to soft RealSR/DRealSR GTs? **Due to the soft GTs**: on DIV2K SeeSR generates 0.6× the GT energy above the LR Nyquist | done |
| H.0b | Controlled test (`data_tools/band_scale.py`, `cluster/band_scale.sh`): scale the frequencies above the LR Nyquist of SeeSR outputs (×0, 0.25, 0.5, 1, 1.5, 2, and a per-octave oracle matching the GT energy), then see which metrics reward more generated detail, on soft (RealSR, DRealSR) and sharp (DIV2K) GTs. **The oracle gives +0.8 dB and LPIPS −0.07 on soft GTs, but only LPIPS −0.017 at −0.18 dB on DIV2K; no-reference metrics reward more detail everywhere** | done |
| H.1 | If H.0 confirms: the same measure on more models (SeeSR, OSEDiff, PiSA-SR, SegESR…), then a calibration of the generated detail driven by the LR (post-hoc or in the decoder of V.2). **Dropped**: even the oracle gains little on sharp GTs | dropped |

