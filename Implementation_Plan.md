# DigitalThread Hackathon — Implementation Plan (Proposal Submitted → Sep 19 Demo)

## Overview
Proposal submitted. This covers everything left to build: synthetic telemetry upgrade, defect classifier, forecasting model upgrade, orientation-characterization strengthening, LLM integration, evaluation, and demo video.

---

## 1. Synthetic Telemetry Data — Noise Realism
**What to do:**
- [X] Modify synthetic data generator so noise scales with progress along the sigmoid degradation curve (near-zero noise early/normal operation, increasing noise approaching/past the fault threshold).
- [X] Decide: continuous noise ramp (smooth, tied to sigmoid progress) vs. abrupt onset (kicks in past inflection point). Continuous is more realistic; abrupt is simpler.
- [X] Regenerate train/test/val telemetry data with new noise model. Keep an old copy for before/after comparison if useful for the demo.
- [X] Use UCR Wafer dataset to inform realistic mean/std baseline values (as stated in proposal).

**Resources:**
- UCR Wafer dataset: available via the UCR Time Series Classification Archive — https://www.timeseriesclassification.com/description.php?Dataset=Wafer (also mirrored on `tslearn` and `sktime` dataset loaders, e.g. `from sktime.datasets import load_UCR_UEA_dataset`)
- No new libraries needed — this is generation logic on top of your existing synthetic data code (numpy/scipy for the sigmoid + noise functions).

---

## 2. Defect Classifier (multi-class, all defect types)
**What to do:**
- [X] Build classifier (recommended: separate model from the anomaly-detection autoencoder, at least initially — encoder can be reused/fine-tuned later if time allows).
- [X] Train on WM-811K across all classes with adequate sample size (Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch; consider excluding Near-full if too small after your resize pipeline).
- [X] Reuse your fixed resize/pad pipeline (nearest-neighbor upsize, max-pool/block-reduce downsize, ceiling-division block sizing) for all classes.
- [X] Standard random train/test split is fine here (no need for the angle-holdout logic — that was specific to the now-dropped invariance experiment).
- [X] Evaluate: accuracy, confusion matrix, per-class precision/recall/F1. Also report on a class-balanced subset given the known imbalance (per your proposal's evaluation section).

**Resources:**
- No new external tooling needed beyond PyTorch/torchvision (already in your stack). A simple CNN classifier (few conv layers + a linear head) is sufficient — no need for anything exotic here; keep this piece low-risk.
- If you want a quick benchmark reference: many public WM-811K notebooks/papers report baseline CNN accuracy in the 90%+ range on the majority classes, lower on rare ones (Scratch, Near-full) — useful context for interpreting your own confusion matrix, not something to copy code from.

---

## 3. Forecasting Model Upgrade (LSTM → modern architecture)
**What to do:**
- [ ] Pick one: **PatchTST** or **N-HiTS** (don't build both — pick based on which library integrates more easily with your existing PyTorch pipeline).
- [ ] Train both old LSTM and new model on the SAME updated (noisy) synthetic telemetry data for a fair comparison.
- [ ] Evaluate: lead-time before fault-threshold crossing (compare how early each model detects the coming fault), forecast error (MSE/MAE).
- [ ] Specifically check: does the new model pick up on the *increasing variance* near the fault region as an early signal, not just the mean trend?

**Resources:**
- **Nixtla's `neuralforecast`** library has ready-to-use implementations of both N-HiTS and PatchTST, with a scikit-learn-like fit/predict interface — https://github.com/Nixtla/neuralforecast
  - Install: `pip install neuralforecast`
  - Docs/examples: https://nixtla.github.io/neuralforecast/
- If you'd rather implement PatchTST more directly in PyTorch (more control, more work): original paper repo — https://github.com/yuqinie98/PatchTST
- N-HiTS original paper repo (if not using neuralforecast's wrapper): https://github.com/Nixtla/neuralforecast (same org maintains both the paper's reference implementation and the packaged library)

---

## 4. Orientation Characterization — Strengthening (Secondary Track)
**What to do:**
- [ ] Already done: PCA angle/linearity extraction, fixed resize pipeline, null-model control, initial Scratch result (n=135, χ²=28.59, p=0.0385).
- [ ] Optional strengthening (if time allows):
  - [ ] Run the same real-vs-null chi-square comparison on a known non-directional class (Center or Donut) as a contrast check.
  - [ ] Re-run at 2–3 different linearity thresholds (e.g. 0.3, 0.4) to check result stability.
- [ ] Wire into LLM report: when classifier flags Scratch, compute angle/linearity, pass into LLM prompt as extra context.

**Resources:**
- No new libraries — this is your existing PCA/eigen-decomposition code (numpy) plus `scipy.stats.chisquare`, already built and working.

---

## 5. LLM Report Agent Updates
**What to do:**
- [ ] Update prompt/context to include: defect type (from new classifier), reconstruction error/anomaly score, and (for Scratch) orientation angle + linearity.
- [ ] Add explicit instruction in the prompt: do not claim confirmed root-cause/tool attribution from orientation alone — frame as a possible diagnostic signal only.
- [ ] Test with a few different defect-type + orientation combinations; check reports read sensibly and don't overreach.

**Resources:**
- Whatever LLM API you're already using in DigitalThread's existing agent (per your GitHub repo) — no new service needed, just prompt/context changes.

---

## 6. Evaluation Compilation
**What to do:**
- [ ] Compile final results into clear, simple visuals: classifier confusion matrix, forecasting lead-time comparison chart (old vs. new model), orientation chi-square result with sample size and p-value stated plainly.
- [ ] Cross-check all numbers against what's actually claimed in your submitted proposal — make sure the demo doesn't contradict or wildly exceed what was proposed without explanation.

---

## 7. Demo Video
**What to do:**
- [ ] Script tightly (per hackathon rubric: Demo Video Clarity is 10%, but a confusing video undercuts everything else too): problem → architecture → headline results → LLM report before/after → close.
- [ ] Show a before/after LLM report example (without vs. with defect-type + orientation context) — concrete, effective demo moment.
- [ ] Keep it concise; no separate written report or slide deck required per the hackathon rules — the video itself is the full final submission.

---

## Suggested Sequencing
1. Synthetic data noise update (foundational — forecasting work depends on it)
2. Classifier (core, standalone, can run in parallel with #3)
3. Forecasting model upgrade (core, standalone, depends on #1)
4. Orientation strengthening + LLM integration (secondary, can run in parallel with #2/#3 once data is stable)
5. Evaluation compilation
6. Demo video (last)

## Reminders
- Sample sizes are thin for some classes — report transparently, don't force conclusions.
- Orientation finding stays secondary/preliminary in framing — don't let it become the headline claim over the classifier/forecasting work.
- Don't let the LLM overclaim causation from orientation — keep the guardrail in the prompt.
- Cross-check final demo claims against what was actually submitted in the proposal.
