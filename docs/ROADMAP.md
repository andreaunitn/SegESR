# SegESR roadmap

Method: **B + C + D**. The evaluation contribution is **A**.

- **B** gives every SAM 2 segment a reliability score.
- **C** controls fidelity versus generation per segment, driven by that score.
- **D** generates in two steps (draft, SAM 2 re-check on the draft, refinement).
- **A** is a segment-level hallucination metric.

The novelty is the per-segment reliability and its use to decide how much to generate in each segment. The closed loop and the two steps are the setting, not the contribution: InstantIR and SegSR already re-estimate conditions during sampling. See the survey notes for the related work: SRSR (NeurIPS'25), SAMSR (IJCAI'25), HoliSDiP, SegSR, InstantIR, MMSR (CVPR'25), CODSR, DTPSR (CVPR'26) and RASS (IJCV'26).

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
| 0.2 | Script that collects all `metrics/*/results_*.json` into one table | todo |
| 0.3 | Align with the SRSR protocol: zero-shot SeeSR must give 25.18 dB on RealSR | todo |

**Gate 0:**
- Does SegESR clearly beat fine-tuned SeeSR at the same number of steps?
- Does the SAM 2 refresh help?
- Where does the README's +2.22 dB come from (dataset, protocol)?

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
