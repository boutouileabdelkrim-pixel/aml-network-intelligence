# Monitoring

Model drift detection using **Evidently**. In production, this would run
daily/weekly and trigger alerts when drift exceeds thresholds.

## Why monitoring matters

An AML model trained on 2022 data may fail on 2024 data because:
- **Data drift**: customer behavior changes (crypto, new payment methods)
- **Concept drift**: fraud patterns evolve (adversarial adaptation)
- **Prediction drift**: output distribution shifts (alert volume explodes)

Regulators require **continuous monitoring** — a model can't be a black box.

## Implementation

`src/monitoring/drift.py` compares:
- **Reference** = training data (last known good distribution)
- **Current** = production data (live traffic, sampled)

Using `evidently.presets.DataDriftPreset`, it detects per-feature drift via:
- **Kolmogorov-Smirnov test** for numeric features
- **Chi-square** for categorical
- **PSI (Population Stability Index)** as fallback

Output:
- `reports/<name>_report.html` — visual interactive report
- `reports/<name>_metrics.json` — numeric summary

## Scenarios tested

`src/monitoring/simulate_drift.py` runs 3 scenarios:

| Scenario | Description | Expected |
|----------|-------------|----------|
| 1. Baseline | Same distribution | No drift |
| 2. Amount shift | Amounts x1.5 | Drift on amount features |
| 3. Velocity spike | Velocity x3-5 | Drift on temporal features |

## Usage

    uv run python -m src.monitoring.simulate_drift

Then open `reports/scenario_*.html` in a browser.

## Production setup (real deployment)

In a real deployment:
1. Log every prediction + feature vector to a database (Postgres, S3)
2. Weekly job pulls the last 7 days as "current" data
3. Compare to the training reference
4. If `drift_ratio > 0.3`, trigger alert + schedule retraining
5. Grafana dashboard for visualization

See Phase 11 for CI/CD automation.
