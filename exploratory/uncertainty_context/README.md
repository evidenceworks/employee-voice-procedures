# Exploratory cross-domain uncertainty analysis

This directory contains the separate analysis described in SI S11. Aggregate results are in `02_CONTRASTS.csv` and `03_DIAGNOSTICS.csv`; `SPECIFICATION.md` records the population, variables, estimator and sensitivity checks. The analysis differs from the main four-configuration model.

Authorized local execution using the original EWCS 2024 V2 respondent data:

```sh
python run_uncertainty_context.py --data /local/ewcs24_dataset_ukda_v2.dta --out local_results
python 05_verify_independent.py --data /local/ewcs24_dataset_ukda_v2.dta --results local_results --out verification.json
```

The public files contain aggregate output and reproducible code; respondent records remain with their data provider.
