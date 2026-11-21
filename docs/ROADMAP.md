# SegESR roadmap

The research directions below share Phase 0. The code, models and results of the closed ones are archived: see [ARCHIVE.md](ARCHIVE.md).

- **Active: Track R (degradation-invariant representation alignment).** A one-step SR model whose U-Net hidden states, on several differently degraded LRs of one HR, are aligned during training with the DINOv2 features of that HR, in place of the HR tags that FiDeSR sees only in training; then the same aligned space as pseudo-labels for real photos without HR. The projection head is dropped at inference. Base: FiDeSR (CVPR'26). See [Track R](#track-r-degradation-invariant-representation-alignment-active).
- **Stopped: Track M (where to generate).** A one-step model whose fidelity-realism setting varies over the image, from a map predicted from the LR, instead of one global setting. Base: FreqOrtho-SR (ECCV'26). Gates M.0 and M.0b failed: a per-tile oracle that sees the GT gains at most +0.3 dB over the global settings at the same MUSIQ, and with a realism budget it falls 0.4–0.75 dB below the default, because the tiles it keeps faithful cost more MUSIQ than they give PSNR. See [Track M](#track-m-spatially-varying-fidelity-realism-setting-stopped).
- **Stopped: BiBandSR.** A one-step SR model trained with losses split at the LR Nyquist limit: fidelity below it, realism above it. Gate O.1 failed: against the control, +0.14–0.30 dB below 1/8 on the test sets (+0.67 dB on the validation crops), with MUSIQ and MANIQA equal at step 500. See [BiBandSR](#bibandsr-one-step-sr-with-losses-split-at-the-lr-nyquist-limit-stopped).
- **Track A (fidelity in the LR-supported band): analysis done, training-free method dropped.** About half of SeeSR's MSE is at frequencies the LR determines, but changes at inference recover only +0.3–0.5 dB. T.0 shows that a model trained for fidelity does not reach that band of the RealSR and DRealSR GTs either (camera mismatch), but does on DIV2K (synthetic degradations, sharp GT: about +2 dB below 1/8), where the post-hoc fusion still costs perception. Continued, with training, in BiBandSR. See [Track A](#track-a-fidelity-in-the-lr-supported-band-analysis-done).
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

## Track R: degradation-invariant representation alignment (active)

One-step SR models on SD 2.1 get the meaning of the scene from text tags. OSEDiff, TADSR and SeeSR run a prompt extractor (RAM or DAPE) on the LR at test. PiSA-SR and FiDeSR give the U-Net the RAM tags of the **GT** in half of the training steps and run with an empty prompt at test (checked in the FiDeSR code: `train_fidesr.py`, `--null_text_ratio 0.5`; `test_fidesr.py`, `validation_prompt = ''`), so the semantics of the HR are privileged information that is gone at test. DAPE (SeeSR) aligns an external tagger on the LR with RAM on the HR; REPA (ICLR'25) aligns a generator's own hidden states with DINOv2. Track R moves the alignment into the generator and makes it invariant to the degradation: the hidden states of the U-Net on K differently degraded LRs of one HR are all aligned with the DINOv2 features of that HR. The projection head is dropped at inference, so the model costs what the base costs; against OSEDiff, TADSR or SeeSR it saves the prompt extractor. A second part (R.1b) uses that aligned space on real photos without an HR: the head of an EMA teacher on the photo is the target of the student on the same photo degraded further. Both target the cause found in T.0 and H.0b, synthetic training degradations against real camera pairs at test.

Base: **FiDeSR** (CVPR'26, Apache-2.0, SD 2.1-base, training code and weights; GT tags in training, DAW-weighted MSE and LPIPS, CSD, LRRB, LFIM at inference). It cannot be retrained from scratch here (200k steps on 2 H100), so every arm is a fine-tuning from its checkpoint with the same steps and data.

Closest work (searches of 2026-11-20 and 2026-11-21, three independent checks):
- **GenDR** (ICLR'26): one-step SD 2.1 SR whose distillation aligns the generator's hidden state, through an MLP, with the features of an encoder (DINOv2 as example) of the HR, cosine, weight 0.1; fixed prompt embeddings at inference. This is arm C below: single-LR alignment is not ours.
- **InstanceRSR** (ICASSP'26): the same alignment with DINOv2 of the HR in a multi-step DiT trained from scratch, one LR per HR.
- SeeSR/DAPE (external tagger aligned LR to HR, kept at inference), RCOD (CLIP vision prompt), GramSR (DINOv3 of the LQ as conditioning), RECON (arXiv 2026, a generator conditioned on its own predicted DINOv2 features, generation only), U-REPA (NeurIPS'25), Latent Harmony (degradation-invariant VAE latent), SDD (AAAI'26, aligns latent features from a low-scale teacher to a high-scale student), PISR (ECCV'20, features of an HR-input teacher distilled to an SR student).
- For R.1b: QualiTeacher (arXiv 2026, EMA teacher on real degraded images, pixel losses, regression networks, no re-degradation), CORUN (NeurIPS'24, real dehazing), LWay (CVPR'24, self-supervised fine-tuning on the test LRs, pixel space), SRTTA (NeurIPS'23) and DCTTA (CVPR'26), test-time adaptation with re-degradation.
- Not found: several degradations of one HR aligned to one target (D), pseudo-labels in an HR-aligned feature space for real photos without HR (R.1b), in any SR or restoration model.

| Step | Goal | Status |
|---|---|---|
| R.0 | Do the premises hold? (a) What the semantics of the HR are worth to FiDeSR at test: official empty prompt (`fidesr`), RAM tags of the LR (`fidesr_ramlr`), RAM tags of the GT as an oracle (`fidesr_ramhr`), on RealSR, DRealSR, DIV2K, with the parameters and time of RAM (`data_tools/fidesr_test.py`, `cluster/onestep_baselines.sh`). (b) On 500 DIV2K HR crops with LRs at three degradation levels: alignment (CKNNA, as in REPA) of the FiDeSR U-Net features with DINOv2 of the HR, per block, and the per-image correlation with LPIPS (`data_tools/feature_alignment.py`, `cluster/feature_alignment.sh`). Inference only | todo |
| R.1 | Does it work? Fine-tuning from FiDeSR, same steps and data, four arms: A the FiDeSR recipe (GT tags in half of the steps, empty prompt at test; control); B no tags; C no tags, alignment on one LR (as GenDR); D no tags, alignment over K = 2 LRs (ours). Layers as U-REPA (middle block, first up block), DINOv2 ViT-B/14, cosine per patch | todo |
| R.1b | Real photos without HR, if R.1 passes. Premise, inference only: on RealSR and DRealSR, are the FiDeSR features of the real LRs less aligned with DINOv2 of the HR than those of synthetic LRs of the same HRs with the same LR PSNR? Then, from D, the same extra steps in three arms: D+ synthetic only; E plus real photos, EMA-teacher consistency in the aligned space (ours); F plus real photos, consistency on the output (LPIPS). Real photos not in any test set (NTIRE 2020 Real-World SR track 2, DPED) | todo |
| R.2 | A method for a paper: longer training; ablations on layers, K, λ_align, encoder (DINOv2, DINOv3); comparison with FiDeSR, GenDR, TADSR, PiSA-SR, OSEDiff, FreqOrtho-SR, and GramSR and RCOD if their code is out; RealLQ250 and RealLR200 too; parameters and time | todo |

**Gate R.0.** Go on to R.1 if (b) holds: the dataset CKNNA drops by at least 20% (relative) from the mild to the strong degradation on at least one block, and on at least one block the per-image alignment (linear CKA of the tokens) correlates with the per-image LPIPS, Spearman ≤ −0.3 on average over the three levels (within each level, so that the degradation itself does not make the correlation). (a) is reported and sets the framing (how much the semantics of the HR are worth at test), not the go.

**Gate R.1.** On at least 2 of the 3 datasets: D is not worse than A (PSNR ≥ −0.1 dB, LPIPS ≤ +0.005, MUSIQ ≥ −0.5, MANIQA ≥ −0.01), and D is better than both B and C (LPIPS at least 0.003 lower or MUSIQ at least 0.5 higher, within the tolerances on the other).

**Gate R.1b.** Premise: on at least one of the two aligned blocks, the real LRs are at least 10% (relative) less aligned than the matched synthetic ones, on both RealSR and DRealSR. Then, on RealSR and on DRealSR: E beats D+ (LPIPS at least 0.003 lower or MUSIQ at least 0.5 higher, within the tolerances of R.1); E beats F in the same way on at least one of the two; on DIV2K E stays within the tolerances of D+.

**Gate R.2.** The final model beats FiDeSR and GenDR in the standard table on at least 2 of the 3 datasets, at the parameters and time of FiDeSR.

## Track M: spatially varying fidelity-realism setting (stopped)

The one-step models of 2026 trade fidelity for realism with global settings: PiSA-SR and FreqOrtho-SR scale their semantic branch by one `lambda_sem` for the whole image; IFCSR and TADSR (CVPR'26) name their global control as a limit. Our steps suggest the best setting changes inside an image: where the LR determines the content (structure, text, edges) fidelity pays, where it does not (fine texture) generation does. The closest work is SROOE (CVPR'23): a GAN generator conditioned by a per-pixel map of its fidelity-realism trade-off, predicted from the LR and trained against a per-pixel LPIPS oracle, with no control left to the user. We differ in three ways: the map scales the semantic branch of a one-step diffusion model that stays frozen (until M.2); the user sets only a floor of realism, and the network decides where to spend it; the comparison is at the same perceptual score. CODSR (CVPR'26) varies the noise per region from a hand-made gradient map fixed at training.

The method, if M.0 or M.0b passes: a light predictor turns the LR into a map of the semantic strength, applied in the latent of a two-branch model. Base: **FreqOrtho-SR** (ECCV'26, Apache-2.0, SD 2.1-base, built on the PiSA-SR code; pixel and semantic branches, `lambda_pix` and `lambda_sem` at inference). Fallback: PiSA-SR. Code and weights in `preset/external/FreqOrtho-SR` and `preset/models/freqortho` (`cluster/setup_bibandsr.sh`).

| Step | Goal | Status |
|---|---|---|
| M.0 | Is there room? FreqOrtho-SR with its default setting and at `lambda_sem` 0, 0.5, 1, 1.5 on RealSR, DRealSR, DIV2K (`cluster/onestep_baselines.sh`, `LAMBDA_SEM`); a per-tile oracle that takes in each 64 or 128-pixel tile the setting closest to the GT, and a control with the same choices at random places (`data_tools/oracle_map.py`, `cluster/oracle_map.sh`); the gap from the curve of the global settings at the same MUSIQ and MANIQA (`data_tools/curve_gap.py`). Inference only | done |
| M.0b | If M.0 fails because the oracle sits at the fidelity end of the curve: an oracle with a realism budget. In each tile it takes the setting that minimises `MSE − μ·lambda_sem`. One price μ for all images, so an image can have all its tiles at 0: it is the optimum, over the dataset, of the M.1 loss below with α = 0, so it bounds M.1. The price is set by a budget β, the mean `lambda_sem` over the tiles of the dataset (as β in M.1): the smallest μ that reaches it, by bisection. β = 0.75, 1, 1.25 on 64-pixel tiles; with the M.0 oracle (μ = 0) and `lambda_sem` = 1.5 (β = 1.5, all tiles) they trace a curve. Control: the same choices at random places, so the same budget (`data_tools/oracle_map.py --budgets`, `cluster/oracle_map.sh` with `BUDGETS`). Inference only | done |
| M.1 | A predictor of the map from the LR, with FreqOrtho-SR frozen, trained with a fidelity loss under a realism budget (below); how much of the oracle gap it keeps | dropped |
| M.2 | The map in the latent with the LoRAs unfrozen, trained end to end with the same loss; comparisons with FreqOrtho-SR, PiSA-SR, CODSR | dropped |

**Gate M.0.** First, the default FreqOrtho-SR should be within about 0.3 dB of its paper (26.27 / 29.08 / 24.35 dB). Go on to M.1 if the 64-pixel oracle is above the curve of the global settings by at least **+0.5 dB of PSNR at the same MUSIQ** and +0.3 dB at the same MANIQA on at least 2 of the 3 datasets, and the random control gains less than half of that. A point outside the range of the curve counts as a fail. The oracle chooses with the GT, so its gain is an upper bound; the control shows how much of it comes from where the choice is made.

**Result (M.0).** The sanity check passes: the default is 26.39 / 29.25 / 24.46 dB, +0.11 to +0.17 from the paper. Gate M.0 fails: at the same MUSIQ the 64-pixel oracle is +0.19 / +0.30 / +0.13 dB above the curve (RealSR / DRealSR / DIV2K), at the same MANIQA +0.23 / +0.33 / +0.17, so only DRealSR reaches +0.3 at the same MANIQA. Where it chooses matters (the random control gains a quarter to a third of it) but the gain is small, and it is measured where the comparison does not interest us: by pixel error the oracle takes `lambda_sem` = 0 in 66–75% of the tiles and 1.5 in 0.5–2%, so it sits at MUSIQ 54 / 49 / 43, near `lambda_sem` = 0 and far from the default (69 / 66 / 69). This is the case M.0b was defined for. See [EXPERIMENTS.md](EXPERIMENTS.md#step-m0-room-for-a-spatially-varying-lambda_sem-in-freqortho-sr).

**Gate M.0b.** At the MUSIQ of the default setting (`lambda_sem` = 1), the curve of the budget oracle is at least **+0.5 dB of PSNR above the default** on at least 2 of the 3 datasets, and the random control with the same budget gains less than half of that.

**Result (M.0b).** Gate M.0b fails, and the shuffled control was cancelled since it could not change the verdict. At the MUSIQ of the default the budget curve is **below** the default: −0.41 / −0.57 / −0.75 dB (RealSR / DRealSR / DIV2K). At the same mean `lambda_sem` (β = 1) the oracle gains +0.52 / +0.38 / +0.40 dB of PSNR, but loses 3.5 / 3.7 / 5.7 MUSIQ and 0.03 to 0.05 MANIQA: the tiles it keeps faithful are smooth, and the no-reference scores, computed on the whole image, drop more than the PSNR rises. Getting the MUSIQ back takes β ≈ 1.25, where the PSNR is already below the default. See [EXPERIMENTS.md](EXPERIMENTS.md#step-m0b-the-oracle-with-a-realism-budget).

**Decision.** Track M stops. M.0b bounds M.1 trained for fidelity, and the bound is below the global setting; a perceptual term in the loss cannot close a gap of that sign. Choosing the tiles by LPIPS, as SROOE does, was not tried: the comparison at the same MUSIQ penalises any faithful region, whatever chooses it. It is the fourth direction (A.0b, T.0, O.1, M.0) to stop at about +0.3 dB over the baselines on these test sets. The step M.1 design below is kept for reference.

**Step M.1 in detail.** A pure fidelity loss would switch the semantic branch off: the M.0 oracle, which picks by pixel error, takes `lambda_sem` = 0 in 66% of the 64-pixel tiles of RealSR and 1.5 in 1%. So a floor β sets the least mean of the map over a batch, and becomes the user's single control; above it, and between the images of a batch, the network places the realism where the loss gains from it.
- Map: Λ = 1.5 · sigmoid(logits + b), with b = max(0, b*) and b* found by bisection so that the mean of Λ over the batch is β. The map stays in [0, 1.5], the range of M.0; if the network already spends more than β, it is left alone; an image of the batch where generating does not pay can stay at Λ ≈ 0. Batches hold crops of different images. β is drawn at random in [0, 1.5] for each batch and given to the predictor. At inference b is fixed, calibrated once on a validation set so that the mean of Λ there is β.
- Diagnostic: how far each image goes above the floor (mean(Λ) − β). With MSE + LPIPS the network should exceed it only where generating also lowers the error (sharp edges, text); if it rarely does, the per-region realism term below is needed.
- Output: ε = ε_pix + Λ ⊙ s, with ε_pix the pixel-branch prediction and s = ε_full − ε_pix the semantic branch, as FreqOrtho-SR already computes them for `lambda_sem` (two UNet passes); then z − ε through the VAE decoder.
- Loss: MSE(SR, GT) + α · LPIPS(SR, GT).
- Input of the predictor: the LR and s (what the semantic branch would add), so it can see where s invents content.
- Cost: FreqOrtho-SR is frozen and ε is linear in Λ, so the gradient goes back through the VAE decoder only; ε_pix and s can be precomputed per crop.
- If the map spends the budget where it costs little fidelity but adds no realism (flat regions), a per-region realism term follows: a patch discriminator or the CSD loss of PiSA-SR.
- Evaluation: sweep β and compare the curve with that of the global `lambda_sem`, as in M.0.

**Gate M.1.** Go on to M.2 if, at the MUSIQ of the default setting, the β curve is at least **+0.3 dB of PSNR above the global curve** on at least 2 of the 3 datasets, and keeps at least half of the gap of the oracle (M.0 or M.0b, whichever passed).

## BiBandSR: one-step SR with losses split at the LR Nyquist limit (stopped)

For ×4 SR the LR holds the spatial frequencies below 1/8 cycle per HR pixel (its Nyquist limit); above it, the content is invented. Our measurements on SeeSR support splitting the losses there:
- above the limit, 73–82% of the error is variance between seeds, so a pixel loss there only pushes towards the average (blur);
- about half of the MSE is below the limit (Track A);
- classifier-free guidance moves the band below the limit (A.0), and distillation losses such as CSD and VSD use it (CFG 7.5).

The model: fidelity losses only below the limit, realism losses (LPIPS, CSD) only above it. In the SD latent the 1/8 cutoff cannot be represented directly (the latent only reaches 1/16 of the HR pixels), so the realism gradient is projected in pixel space through the decoder.

Base: **PiSA-SR** (CVPR'25, Apache-2.0, SD 2.1-base). It already separates fidelity and semantics, but in parameters (a pixel LoRA trained with L2, a semantic LoRA with LPIPS + CSD, both on all frequencies). Its CSD loss needs one frozen UNet only, lighter than the VSD of OSEDiff. **OSEDiff** (Apache-2.0) is a second baseline, inference only.

Environment: `~/venvs/bibandsr` (`requirements-bibandsr.txt`, `cluster/setup_bibandsr.sh`), on the same container: PiSA-SR and OSEDiff need diffusers 0.25 and peft 0.9. Downloaded code and weights go to `preset/` (`preset/external/`, `preset/models/`).

| Step | Goal | Status |
|---|---|---|
| O.0 | PiSA-SR and OSEDiff with their official weights, scored with our protocol on RealSR, DRealSR and DIV2K: metrics, band errors and energy (`data_tools/collect_sr.py`, `cluster/onestep_baselines.sh`). **Gate passes**: 41–66% of the MSE below 1/8; both are worse than SeeSR in the lowest octave on every dataset | done |
| O.1 | Two identical short fine-tunings from the official PiSA-SR weights, on our data and hardware: the original losses (control) and the split losses. Checkpoints and automatic resubmission past the 1-day limit (`bibandsr/train.py`, `bibandsr/bands.py`, `cluster/bibandsr_train.sh`). **Gate fails**: the split run gains +0.30 / +0.27 / +0.16 dB below 1/8 at step 2000 (MUSIQ about −1, MANIQA about −0.01) and +0.18 / +0.27 / +0.14 dB at step 500 (MUSIQ and MANIQA equal); the control alone moves PiSA-SR towards fidelity (+0.4–0.6 dB, MUSIQ −3 to −4) | done |
| O.2 | If O.1 shows a gain: the split losses behind flags in a full training; ablations (fidelity split only, realism split only, cutoff 1/16 or 1/8) | dropped |
| O.3 | The same on OSEDiff; comparisons (PiSA-SR, OSEDiff, SeeSR, others); writing | dropped |

**Gate O.0.** Go on if, in the one-step models too, the band below 1/8 holds a large part of the error (at least about 30% of the MSE), as in SeeSR.

**Step O.1 in detail.** As in the second stage of the official training, only the semantic LoRA is trained (the pixel LoRA stays frozen), on the LSDIR training subset with the Real-ESRGAN degradations of PiSA-SR, RAM prompts from the GT, the official settings (lr 5e-5, L2 + 2 LPIPS + CSD with CFG 7.5, timesteps 0.02–0.5) and 8 samples per update. The two runs differ only in the losses:
- **control** (`--loss pisasr`): the official losses, all on every frequency;
- **split** (`--loss split`): L2 only below 1/8; LPIPS and CSD with their gradient only above 1/8 (stop-gradient on the low band). The latent step of CSD is moved to pixel space through the decoder by a finite difference, high-passed, and scaled so that without the filter its gradient on the latent has the norm of the latent CSD's (calibrated on the first batches).

Both runs first measure, on the same batches, the share of the CSD and LPIPS gradients below 1/8 (`calibration.json`): how much of the realism losses acts on the band the LR determines. During training, 32 held-out crops give PSNR below and above 1/8 and LPIPS (`val.jsonl`); at the end the last checkpoint is scored on RealSR, DRealSR and DIV2K. A short smoke run first measures speed and memory, to set the number of steps.

**Gate O.1.** Go on to O.2 if, against the control at the same step, the split run gains at least **+0.3 dB of PSNR below 1/8** on RealSR, DRealSR and DIV2K and some total PSNR, with LPIPS no worse than +0.005 and MUSIQ and MANIQA within about 1 point (0.01 for MANIQA). If the control alone moves the metrics, the comparison is with the control, not with the official weights.

**Result.** Gate O.1 fails. The gain is consistent in sign (6 of 6 comparisons, LPIPS never worse) and sits below 1/8 as designed, but it is about a third of the gain on the validation crops, which share the training degradations. See [EXPERIMENTS.md](EXPERIMENTS.md#step-o1-split-losses-against-the-official-ones).

## Track A: fidelity in the LR-supported band (analysis done)

For ×4 SR, the frequencies below 1/8 cycle per HR pixel are determined by the LR. The band PSNRs of V.1 put about 43% (RealSR), 54% (DRealSR) and 41% (DIV2K) of SeeSR's MSE there, and 15–24% in the lowest octave [0, 1/16) alone. The VAE reconstructs these bands almost exactly (47–50 dB in the lowest one), so the error comes from sampling. SegESR and fine-tuned SeeSR lose fidelity mostly there too.

Goal: find what moves the LR-supported band during sampling and reduce it without training, for any model, without knowing the degradation. Related: FiDeSR (CVPR'26: low-frequency consistency in one-step models, with training); ILVR and DDNM (low band from the LR, with a known degradation); the wavelet color fix of StableSR.

| Step | Goal | Status |
|---|---|---|
| A.0 | Diagnosis on zero-shot SeeSR (RealSR + DRealSR): guidance 1 / 3 / 5.5; color fix adain / wavelet / none; 3 seeds, split of each band error into a seed part and a systematic part (`data_tools/seed_spread.py`); oracle with the band below 1/16 or 1/8 from the GT and a fusion with the mean of the seeds (`band_scale.py --low_from`, `cluster/low_band.sh`). **Gate passes with a small margin**: guidance 1 gives +0.67 / +0.69 dB below 1/8, the mean of 3 seeds +0.61 / +0.78; about 80% of the error is systematic | done |
| A.0b | Before changing the sampler: the band below 1/16 or 1/8 of guidance 1 combined with the band above of guidance 5.5 (what a frequency-split guidance could give), and the band errors of the bicubic LR (`data_tools/bicubic.py`, `cluster/low_band_proxy.sh`). Go on to A.1 if the combination gains at least about 0.6 dB of PSNR and keeps MUSIQ and MANIQA within about 1 point of SeeSR. **Fails**: +0.26 / +0.36 dB; with the lowest octave from the bicubic LR +0.48 / +0.50 dB, but LPIPS, MANIQA and FID get worse | done |
| A.1 | Training-free method (frequency-split guidance, low band from the seeds or the LR). **Dropped**: at most about +0.3–0.5 dB (A.0b) | dropped |
| T.0 | Feasibility of a trained reconstruction, without training: the band below 1/16 or 1/8 of SeeSR taken from regression SR models trained with a pixel loss on realistic degradations (RealESRNet_x4plus, BSRNet; `data_tools/regression_sr.py`, `data_tools/rrdbnet.py`, `cluster/regression_fusion.sh`). **Fails on RealSR and DRealSR**: at most +0.28 / +0.09 dB, with worse FID and MANIQA. On DIV2K +0.79 dB (RealESRNet below 1/8), still below +1.0 and with FID +12 | done |
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

