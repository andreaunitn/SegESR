# SegESR roadmap

Two research directions share Phase 0.

- **Active: Track V (VAE).** A decoder that can be plugged into any latent SR model, where each pixel uses as much LR evidence as the LR actually holds, plus an analysis of how much of the SR error comes from the VAE. See [Track V](#track-v-evidence-aware-decoder-active). Stop criteria and the fallback track H are at the end of the file.
- **On hold: the segmentation track (Phases 1–5 below).**

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
| 0.1 | Test SegESR (with and without SAM 2 refresh) and fine-tuned SeeSR at `checkpoint-15000` on RealSR + DRealSR | doing |
| 0.2 | Script that collects all `metrics/*/results_*.json` into one table (`data_tools/results_table.py`) | done |
| 0.3 | Align with the SRSR protocol: zero-shot SeeSR must give 25.18 dB on RealSR | todo |
| 0.4 | Runtime per image in `test.py` (only measurable during generation), plus TOPIQ and LIQE in `metrics.py` (already in pyiqa 0.1.13). A-FINE needs a newer pyiqa: added before the final comparisons, recomputing every metric with the same version | todo |

**Gate 0:**
- Does SegESR clearly beat fine-tuned SeeSR at the same number of steps?
- Does the SAM 2 refresh help?
- Where does the README's +2.22 dB come from (dataset, protocol)?

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

## Track V: evidence-aware decoder (active)

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
| V.0 | VAE ceiling on RealSR and DRealSR: decoder(encoder(GT)) and decoder(best latent), scored like SR outputs (`data_tools/vae_ceiling.py`, `cluster/vae_ceiling.sh`). Compare with the SeeSR and SegESR outputs | doing |
| V.1 | Fine-structure benchmark: error per frequency band, edge F-score, OCR on small text (as PGSR, TextOCR). Bands and edges: `data_tools/fine_structure.py`, `cluster/fine_structure.sh`. **Bands and edges fail the gate** (VAE share of the error 9–13%); OCR postponed | done |
| V.2 | Baseline decoder adapter: LR features injected uniformly into the frozen SD VAE decoder (CFW- and PGSR-like), trained on latents of a frozen SR model | todo |
| V.3 | Evidence-aware gating: a per-pixel gate predicted from the LR (how informative it is locally), with a global scale for inference-time control | todo |
| V.4 | Plug-in test: the same decoder on other frozen models (SeeSR, SegESR, OSEDiff, PiSA-SR), with and without light fine-tuning | todo |
| V.5 | Baselines (StableSR CFW, GuideSR, TVT, PGSR on SD 2 if its code allows), ablations, writing | todo |

**Gate V.0 (decoder(encoder(GT)) done, best latent running).**
- The VAE reconstruction of the GT reaches 35.39 dB / LPIPS 0.0365 on RealSR and 39.78 dB / 0.0335 on DRealSR.
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

**Key fact.** For ×4 SR the LR holds frequencies only below 1/8 cycle per HR pixel, so everything above it is generated. V.1 shows the models generate 6–12× the GT's energy in the highest octave, more after fine-tuning and with SegESR.

| Step | Goal | Status |
|---|---|---|
| H.0a | V.1 metrics on DIV2K (sharp GT): does the excess of generated energy hold, or is it due to soft RealSR/DRealSR GTs? | doing |
| H.0b | Controlled test (`data_tools/band_scale.py`, `cluster/band_scale.sh`): scale the frequencies above the LR Nyquist of SeeSR outputs (×0, 0.25, 0.5, 1, 1.5, 2, and a per-octave oracle matching the GT energy), then see which metrics reward more generated detail | doing |
| H.1 | If H.0 confirms: the same measure on more models (SeeSR, OSEDiff, PiSA-SR, SegESR…), then a calibration of the generated detail driven by the LR (post-hoc or in the decoder of V.2) | todo |

