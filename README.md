# Employee voice and procedural clarity

Reproducible aggregate results and analysis code for an EWCS 2024 (second edition) study of 6,548 employees in the EU27. The study compares four configurations of reported employee voice with three responses concerning a clear procedure for challenging computer-mediated decisions.

[Run the aggregate verification in Google Colab](https://colab.research.google.com/github/evidenceworks/employee-voice-procedures/blob/main/notebooks/verify_results.ipynb)

In Colab, select **Runtime → Run all**. The notebook retrieves the public repository, checks the released aggregate tables and inferential records, reproduces the seven covariance-derived comparisons and provides an aggregate export. Figure generation is optional. No upload of the repository or respondent data is required.

For local aggregate checks:

```sh
python code/verify_scientific_files.py
python code/verify_public_results.py --out ../aggregate-verification
python -m unittest discover -s tests -v
```

Reestimating the survey models requires authorized access to [EWCS 2024 V2, UK Data Service study 9511](https://doi.org/10.5255/UKDA-SN-9511-2):

```sh
python code/reproduce.py --source-zip ../UKDS_STATA.zip --out ../ewcs-reproduction
```

Variable definitions and the survey design are documented in `documentation/`. Code is released under [MIT](LICENSE); rights to the original data are described in [third-party.md](third-party.md).
