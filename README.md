# Employee voice and reported procedural clarity

Code and aggregate results for the analysis of four employee-voice configurations and three procedural responses among 6,548 eligible employees in the EU27, using the European Working Conditions Survey 2024, second edition.

The survey data are available from the [UK Data Service, study 9511](https://doi.org/10.5255/UKDA-SN-9511-2) under its access conditions. Respondent records are not distributed here.

## Verify published results

[Open in Colab](https://colab.research.google.com/github/evidenceworks/employee-voice-procedures/blob/main/notebooks/verify_results.ipynb)

The notebook checks the released aggregate results, displays employee-voice profiles and sensitivity estimates and downloads `employee-voice-results.zip` with three CSV tables and a verification record. It runs without respondent-level data.

## Reproduce the analysis

Use Python 3.12 with the dependencies in `requirements.txt`. After obtaining the authorized Stata survey archive, run:

```sh
python code/reproduce.py --source-zip "../UKDS_STATA.zip" --out "../ewcs-reproduction"
```

The program reproduces the study population, model contrasts, sensitivity estimates and aggregate tables. It expects the second-edition UKDS Stata ZIP. The output directory must be new or empty. Keep its respondent-level intermediate files private.

The released aggregates can also be checked locally:

```sh
python code/check_aggregates.py --results results
python code/verify_scientific_files.py
```

Figures can be generated from the released aggregates with `python code/plot_figures.py --tables-dir results --baseline-csv results/BASELINE_Q57.csv --out ../ewcs-figures`.

The code, numerical results and variable definitions are organized under `code/`, `results/` and `documentation/`. The [MIT license](LICENSE) covers original code. See [third-party rights](third-party.md) for source conditions and public documentation.
