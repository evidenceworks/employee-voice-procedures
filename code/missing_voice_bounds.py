#!/usr/bin/env python3
"""Conservative unweighted bounds from released counts, without respondent data.

Recorded Q57 answers are fixed. Every voice-incomplete case may be assigned to
any Q80a x Q80c configuration. This relaxes restrictions from a partially observed
voice item, so the resulting intervals are outer bounds, not confidence intervals.
They do not bound survey-weighted or covariate-adjusted estimates.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

GROUPS = ('neither', 'representation_only', 'meetings_only', 'both')

def read_counts(results: Path) -> tuple[dict[str, tuple[int, int]], int, int]:
    cells: dict[str, dict[str, int]] = {g: {} for g in GROUPS}
    totals: dict[str, int] = {}
    with (results / 'T1_Procedure_by_voice.csv').open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            g, state = row['group'], row['outcome']
            if g not in cells or state not in ('1', '2', '3'):
                raise ValueError('Unexpected configuration or response code')
            if state in cells[g]:
                raise ValueError('Duplicate configuration-response cell')
            cells[g][state] = int(row['cell_n'])
            n = int(row['group_n'])
            if g in totals and totals[g] != n:
                raise ValueError('Inconsistent group denominators')
            totals[g] = n
    counts = {}
    for g in GROUPS:
        if set(cells[g]) != {'1', '2', '3'} or sum(cells[g].values()) != totals[g]:
            raise ValueError('Incomplete response partition')
        counts[g] = (cells[g]['1'], totals[g])
    missing = {}
    with (results / 'voice_item_response_counts.csv').open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            state = row['q57_response']
            total, retained, excluded = (int(row[k]) for k in (
                'routed_q57_valid_n', 'both_voice_items_valid_n', 'voice_incomplete_n'))
            if total != retained + excluded:
                raise ValueError('Sample-flow counts do not reconcile')
            missing[state] = excluded
    if set(missing) != {'Yes', 'No', 'Does not know'}:
        raise ValueError('Unexpected missing-case response labels')
    return counts, missing['Yes'], missing['No'] + missing['Does not know']

def bounds(y: int, n: int, missing_yes: int, missing_other: int) -> tuple[float, float]:
    """Extremes of the affirmative share under unrestricted missing allocations."""
    return y / (n + missing_other), (y + missing_yes) / (n + missing_yes)

def compute(results: Path) -> list[dict[str, str | int | float]]:
    counts, a, b = read_counts(results)
    rows = []
    for g in GROUPS:
        y, n = counts[g]
        low, high = bounds(y, n, a, b)
        rows.append(dict(estimand=f'{g}: negative_or_uncertain_share',
            observed=100 * (n-y)/n, lower=100*(1-high), upper=100*(1-low),
            units='percent', weighting='unweighted',
            lower_formula=f'100*({n}-{y})/({n}+{a})',
            upper_formula=f'100*({n}-{y}+{b})/({n}+{b})'))
    y0, n0 = counts['neither']
    for g in GROUPS[1:]:
        y, n = counts[g]
        # The two endpoint allocations use disjoint response sets and are feasible
        # in the relaxed allocation space for each contrast considered separately.
        lower = y/(n+b) - (y0+a)/(n0+a)
        upper = (y+a)/(n+a) - y0/(n0+b)
        rows.append(dict(estimand=f'{g}_minus_neither: affirmative_difference',
            observed=100*(y/n-y0/n0), lower=100*lower, upper=100*upper,
            units='percentage points', weighting='unweighted',
            lower_formula=f'100*({y}/({n}+{b})-({y0}+{a})/({n0}+{a}))',
            upper_formula=f'100*(({y}+{a})/({n}+{a})-{y0}/({n0}+{b}))'))
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--out', type=Path, default=Path('results/missing_voice_allocation_bounds.csv'))
    args = parser.parse_args()
    rows = compute(args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {len(rows)} aggregate-only bounds to {args.out}')

if __name__ == '__main__':
    main()
