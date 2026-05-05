# FTIR Sports ML — Study 1

Supervised machine learning for sport group discrimination (sedentary / football / ultrarunning)
across five biological matrices (capillary blood, plasma, saliva, serum, urine) using FTIR spectroscopy.

## License

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

© 2024–2025 Pedro Afonso Valente, University of Coimbra.
Licensed under [CC BY-NC-ND 4.0](LICENSE) — share with attribution, no commercial use, no derivatives.

This code accompanies the manuscript:

> "Discrimination of Sedentary and Athletic Profiles via FTIR-Based Digital Fingerprints
> combined with Supervised Machine Learning"
> P.A. Valente et al., submitted to *Computer Methods and Programs in Biomedicine*

If you use this code or data, please cite the manuscript above.

---

## Quick start (Docker — no installation needed)

**Prerequisite**: [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

```bash
git clone https://github.com/pedroasvalente/ftir-sports-ml.git
cd ftir-sports-ml
docker compose up streamlit
```

Open **http://localhost:8501** in your browser.

The dashboard lets you explore spectra, PCA, PLS-DA scores, VIP scores, and ML results
for each biological matrix without running any training yourself.

---

## Experiment tracking

All training runs are logged publicly on DagsHub — no account needed:

**→ [dagshub.com/pedroasvalente/ftir-sports-ml/experiments](https://dagshub.com/pedroasvalente/ftir-sports-ml/experiments)**

---

## Running the training yourself

> Only needed if you want to reproduce or extend the results.
> Requires a DagsHub token (see `.env.example`).

```bash
# 1. Copy and fill in the environment file
cp .env.example .env
# edit .env: add your DAGSHUB_USER_TOKEN

# 2. Build the container
docker compose build

# 3. Quick test — one matrix, one model (~2 min)
docker compose run --rm train experiments/configs/study1_quick_test.json

# 4. Full Study 1 — all matrices, all models (~hours)
docker compose run --rm train experiments/configs/study1_group_fam.json
```

Results are saved to `results/` and logged to DagsHub automatically.

### After a new training run — updating the dashboard diagnostics

The **Model Diagnostics** page (confusion matrices, ROC curves, VIP scores) reads
diagnostic data from DagsHub artifacts. These are uploaded automatically at the end
of each training run. If you need to upload them manually for an existing run:

```bash
# Replace <run_slug> with the run_name in your experiment config (e.g. study1_group_fam_v2)
.venv/bin/python - <<'EOF'
from pathlib import Path
from ftir.models.training import _log_artifacts_to_mlflow
_log_artifacts_to_mlflow(
    "<run_slug>",
    Path("results/<run_slug>"),
    "FTIR Study 1 - Sport Group Discrimination",
)
EOF
```

After the upload you can verify the artifacts at:
**→ [dagshub.com/pedroasvalente/ftir-sports-ml/experiments](https://dagshub.com/pedroasvalente/ftir-sports-ml/experiments)**
under the experiment **"FTIR Study 1 - Sport Group Discrimination | diagnostics"**.

---

## Study design

| | Sedentary (S) | Football (F) | Ultrarunning (U) |
|---|---|---|---|
| Participants | 46 | 48 | 58 |
| Timepoints | 3 (baseline, +2 m, +4 m) | 1 | 1 |

- **5 biological matrices**: always analysed independently, never combined
- **Target variable**: `group_fam` (S / F / U)
- **Train/test split**: 70/30, person-aware (no data leakage across timepoints)

## Methodological notes

- **Data leakage fix**: `StratifiedGroupKFold` with `person_code` as group key — the same
  individual never appears in both train and test sets
- **Feature importance**: VIP scores from PLS-DA (replaces back-projection used in prior work)
- **Standardised split**: 70/30 across all matrices and conditions
- **SMOTE**: applied to training set only, after splitting and scaling
- **Models**: Random Forest, MLP, Decision Tree, XGBoost — GridSearchCV + BayesSearchCV

## Repository structure

```
data/                  FTIR dataset (001_3_cleaned_FTIR.csv, 1035 samples × 764 wavenumbers)
src/ftir/
├── data/              loader, column config, person-aware splits
├── preprocessing/     scale → PLS-DA → SMOTE pipeline
├── reduction/         PLSDA (with VIP scores), PCA wrapper
├── models/            model configs, training loop, evaluation
├── analysis/          confounder tests (ANCOVA), cross-fluid effect sizes
└── visualization/     spectra mean±SD, confusion matrix, ROC, VIP plots
app/                   Streamlit dashboard (spectra · PCA · PLS-DA · results)
experiments/configs/   JSON experiment configs
```
