# Decision log: GreenOps Phase 2

Team 18, AI252TA (MLOps). 2026-11-06. One entry per decision that changed what we built: what we chose, why, what we rejected, and the evidence. Companion files: `docs/model_card.md`, `docs/model_risk.md`, `docs/data_card.md`.

## Models and evaluation

**D1. Split by date, choose on validation only.**
Train to 2023-12-31, validation 2024, test 2025 onward. Models, thresholds and the LSTM are chosen on validation; test is reported, never used to choose.
Rejected: random split (leaks neighbouring days), choosing on test.
Evidence: the one time test disagreed (ridge 0.5496 beat hgb 0.5751 on sunlight test) we kept hgb and said so. Switching after seeing test would make the test score meaningless.

**D2. Beat the simple baselines first.**
Every task is scored against naive, seasonal and persistence ("tomorrow = today"); low-wind against "yesterday's flag". A model that does not beat persistence is not promoted (gate G1, G5).
Evidence: sunlight 0.5369 vs persistence 0.5731 (6.3% better); hub wind 0.7305 vs 0.7492 (2.5%); low-wind PR-AUC 0.6500 vs 0.4992. The margins are small and we say so in the model card.

**D3. Champions: hgb for sunlight, ridge for hub wind, logistic for low-wind.**
Chosen on validation among ridge and hgb (and logistic and hgb for the classifier). Low-wind uses PR-AUC because the event is rare, and its alert threshold (0.226) is picked on validation F1.
Rejected: accuracy for low-wind (a model that never alerts scores well).

**D4. LSTM (TensorFlow) is a shadow, not the champion.**
Asked by the evaluator to consider LSTM and TensorFlow. Rule fixed before running: promote only if the 5-seed mean validation MAE is at least 2% below the best sklearn model. Window of 7 days, one LSTM layer, L1 loss, early stopping on validation, scaling fitted on training days.
Result: sunlight 0.5201 (std 0.0007) vs hgb 0.5369, 3.1% better; hub wind 0.7096 (std 0.0049) vs ridge 0.7305, 2.9% better. Both pass the rule.
Decision: serve it beside the champion (`compare=true`), do not replace the champion yet. Why: it needs TensorFlow in the serving image (large and slower to start), it has no device-level comparison yet (lamp, heater and turbine error), and a promoted model must also pass the gates.
Revisit: when the TensorFlow image size is acceptable and the device-level comparison is done.

## Packaging and serving

**D5. skops with SHA-256 checks, not pickle or joblib.**
Pickle-style files can run code when loaded. skops loads only trusted types; every model file and the weather snapshot are checksum-verified; the scikit-learn minor version must match; the service refuses to start otherwise.
Evidence: `tests/devices/test_bundle.py`; risk R15.

**D6. One baked bundle, replay mode, no live weather call.**
The image contains the models and a weather snapshot (2019-10-30 to 2026-09-30). The API forecasts the day after any chosen as-of date.
Why: NASA POWER publishes with a delay of days (about 2 to 3 for meteorology, 5 to 7 for solar), so a live call could not give a clean "tomorrow"; the demo must work without network access; every answer is reproducible.
Cost: dates outside the snapshot return 409. Live ingestion is future work.

**D7. Missing weather gives a flagged persistence fallback, not an error.**
`weather_used.fallback = true` and the front end says so. Gate G6 blanks a day at every site and checks it.
Rejected: HTTP 500 or an empty answer (a demo that crashes on a gap is worse than a plainly labelled simple estimate).

**D8. The bundle is built once and pinned; CI and deploy never retrain.**
skops stores a timestamp, so retraining always produces different file bytes (this surprised us during integration). The pinned bundle in `dvc.lock` is pulled by CI and baked into the image; the image needs no secrets at runtime.
Rejected: regenerating the bundle in CI (a different artifact every build).

## Tracking, registry and gates

**D9. TensorBoard for experiments, MLflow for the registry only.**
TensorBoard holds the leaderboard of every model and the LSTM runs (it matches the TensorFlow request); MLflow holds versions and aliases (champion, challenger, previous), one registered model per task.
Note: the TensorBoard HParams table showed blank cells on one Windows setup and we could not explain it; the TEXT tab leaderboard and the charts are the reliable views and are what we show.
Rejected: MLflow stages (deprecated in favour of aliases).

**D10. A model version is identified by algorithm, inputs and metrics, not by file hash.**
Because of the skops timestamp (D8), the same trained model has different bytes each time. A fingerprint avoids duplicate versions; the file SHA-256 is still stored and checked as the integrity tag.

**D11. Seven quality gates (G1 to G7) decide promotion, with thresholds fixed before the real run.**
Beat persistence; no regression (within 2% of the champion); sane outputs; per-site error under 1.5x pooled; low-wind beats yesterday's flag; robustness (missing day, 5% noise under 10% change); honesty (claims recomputed, leak checks).
We chose the numbers before seeing the real data and committed to report a failure, not to loosen a gate. All gates passed for the v1 champions (`results/gates/report.json`).
Evidence: `gates.py`, `tests/devices/test_gates.py` (30 tests, including a deliberately leaky forecast path that must be caught).

**D12. A bad-candidate generator proves the gates work.**
`gates make-bad` builds three broken sunlight models: shuffled labels (loses to the champion), a kWh to Wh unit bug (out-of-range output) and inflated metric claims (passes a claims-only check, caught by recomputing). Each is blocked by the registry. This is the "blocked model" moment of the demo.

## Process and scope

**D13. Phase 1 statewide models are not served.**
The Phase 1 training code registers models by test MAE, which breaks D1. Only the device pipeline models (D3) are in the bundle. Phase 1 stages stay in the repository as history; we do not run a bare `dvc repro`.

**D14. Integration through `phase2`, one reviewed merge to `main`.**
Each person works on a branch, merges into `phase2` by pull request, and one pull request moves `phase2` to `main` followed by the tag `v1.0-phase2`. Publishing the image only happens from `main` or a manual run; the gate in the workflow is not weakened for feature branches.

**D15. Awareness product, not a control system.**
The front end says what tomorrow's weather means for a lamp, a water heater and a small turbine, one day ahead, for eight sites. It shows the data mode, the model version and the margin over "tomorrow = today". It is not for grid, safety or money decisions (risk R18).
