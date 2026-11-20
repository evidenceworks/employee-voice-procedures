"""Recover response-state comparisons from released EWCS model contrasts.

Run:
  python response_state_comparisons.py --workbook Data_S1.xlsx --out response_comparisons

Alternatively use --results /path/to/repository/results instead of --workbook.

Uses published aggregate estimates only. No respondent records are read.
All intervals are pointwise and exploratory. The three response indicators
partition the same sample with identical design matrices, weights and linear
variance operators. This identifies the cross-response covariance for each
contrast through the zero-sum coefficient identity.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import numpy as np
from scipy.stats import t, f

DF = 2455
OUTCOMES = ['procedure_yes', 'procedure_no', 'procedure_unknown']
CONTRASTS = [
    'representation_only - neither',
    'meetings_only - neither',
    'both - neither',
    'both - meetings_only',
    'both - representation_only',
    'meetings_only - representation_only',
    'additive_interaction',
]


def load_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def source_lookup(results=None, workbook=None):
    if workbook is not None:
        from openpyxl import load_workbook
        book = load_workbook(workbook, read_only=True, data_only=True)
        sheet_rows = iter(book['Model_contrasts'].values)
        fields = next(sheet_rows)
        records = [dict(zip(fields, row)) for row in sheet_rows]
        lookup = {(r['outcome'], r['contrast']): r for r in records
                  if r['model'] in ['A_' + name for name in OUTCOMES] + ['Derived_covariance']}
        book.close()
        return lookup
    published = load_csv(results / 'T7_Model_contrasts.csv')
    derived = load_csv(results / 'BASELINE_Q57.csv')
    lookup = {(r['outcome'], r['contrast']): r for r in published
              if r['model'] in ['A_' + name for name in OUTCOMES]}
    lookup.update({(r['outcome'], r['contrast']): r for r in derived
                   if r['contrast'] == 'both - neither'})
    return lookup


def calculate(lookup):
    rows = []
    internal_joint = []
    for contrast in CONTRASTS:
        source = [lookup[(outcome, contrast)] for outcome in OUTCOMES]
        b_yes, b_no, b_unknown = [float(r['estimate_pp']) for r in source]
        v_yes, v_no, v_unknown = [float(r['se_pp']) ** 2 for r in source]
        if abs(b_yes + b_no + b_unknown) > 3e-8:
            raise ValueError(f'Response partition fails: {contrast}')
        cov_no_unknown = (v_yes - v_no - v_unknown) / 2
        V = np.array([[v_no, cov_no_unknown], [cov_no_unknown, v_unknown]])
        eigenvalues = np.linalg.eigvalsh(V)
        if min(eigenvalues) <= 0:
            raise ValueError(f'Invalid recovered covariance: {contrast}')
        estimate = b_unknown - b_no
        variance = v_unknown + v_no - 2 * cov_no_unknown
        standard_error = math.sqrt(variance)
        critical = t.ppf(.975, DF)
        rows.append({
            'configuration_contrast': contrast,
            'response_comparison': 'does not know minus no clear procedure',
            'estimate_pp': estimate,
            'se_pp': standard_error,
            'lo95_pp': estimate - critical * standard_error,
            'hi95_pp': estimate + critical * standard_error,
            'p_pointwise': float(2 * t.sf(abs(estimate / standard_error), DF)),
            'covariance_no_unknown_pp2': cov_no_unknown,
            'n': 6548,
            'df': DF,
            'scope': 'Exploratory aggregate derivation; pointwise inference',
        })
        b = np.array([b_no, b_unknown])
        F = float(b @ np.linalg.solve(V, b) / 2)
        internal_joint.append({
            'configuration_contrast': contrast,
            'F_two_response_states': F,
            'df_numerator': 2,
            'df_denominator': DF,
            'p_pointwise': float(f.sf(F, 2, DF)),
            'interpretation': 'Exploratory joint departure of the response distribution; no mechanism identified',
        })
    return rows, internal_joint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--results', type=Path)
    inputs.add_argument('--workbook', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    rows, joint = calculate(source_lookup(results=args.results, workbook=args.workbook))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'response_state_comparisons.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        'status': 'PASS',
        'input_scope': 'Released aggregate estimates and standard errors, rounded to eight decimal places',
        'microdata_reestimated': False,
        'formula': 'Cov(no, unknown) = [Var(yes) - Var(no) - Var(unknown)] / 2',
        'assumption': 'Identical analytic sample, X, weights and linear covariance operator for the three partitioning outcomes',
        'rows': len(rows),
        'joint_tests_internal_audit_only': joint,
    }
    (args.out / 'response_comparison_receipt.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'comparisons': len(rows), 'respondent_records_read': 0}))


if __name__ == '__main__':
    main()
