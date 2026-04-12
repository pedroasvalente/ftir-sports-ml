# Data

## Processed dataset

`001_3_cleaned_FTIR.csv` — the processed spectral dataset used for all analyses.
This file is tracked in git and is the single source of truth for the project.

It was produced by the preprocessing pipeline (baseline correction → L2 normalisation → outlier removal)
applied to the raw FTIR acquisitions. It is **never modified** — all analysis reads it as-is.

## Raw data

Raw JSON files (spectral acquisitions) are **not** included in this repository due to size.
They are archived separately and are not needed to reproduce the analyses.

## Matrices

Five biological matrices, always analysed independently:

| Code | Matrix |
|------|--------|
| `CAPILAR` | Capillary blood |
| `PLASMA` | Blood plasma |
| `SALIVA` | Saliva |
| `SERUM` | Blood serum |
| `URINE` | Urine |

## Study groups

| Code | Group | Timepoints |
|------|-------|-----------|
| `S` | Sedentary | 3 (baseline, T2 at 2 months, T3 at 4 months) |
| `F` | Football athletes | 1 |
| `U` | Ultrarunning athletes | 1 |
