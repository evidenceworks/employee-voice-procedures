"""Summarize item completeness from the released unweighted Q57 counts.

Usage:
    python code/voice_item_response.py --results results --out /tmp/voice-check

The denominator counts are from the original EWCS 2024 V2 sample flow. The
complete-response counts are cross-checked against the published 4x3 table.
No respondent-level records or new models are needed.
"""
from __future__ import annotations
import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path


def load(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def calculate(results: Path):
    denominators = load(results / "voice_response_denominators.csv")
    assert [r["domain"] for r in denominators] == [
        "Q57-valid eligible employees", "Q57-valid, voice-complete employees"
    ]
    fields = ("procedure_yes_n", "procedure_no_n", "procedure_unknown_n")
    totals, observed = [[int(r[f]) for f in fields] for r in denominators]
    assert sum(totals) == 6988 and sum(observed) == 6548
    counts = load(results / "T1_Procedure_by_voice.csv")
    for idx, observed_n in enumerate(observed, 1):
        assert sum(int(r["cell_n"]) for r in counts if int(r["outcome"]) == idx) == observed_n
    excluded = [all_n - included_n for all_n, included_n in zip(totals, observed)]
    assert excluded == [210, 74, 156]
    assert sum(excluded) == 440
    output = []
    for label, n_total, n_kept, n_excluded in zip(("Yes", "No", "Does not know"), totals, observed, excluded):
        output.append({
            "q57_response": label,
            "routed_q57_valid_n": n_total,
            "both_voice_items_valid_n": n_kept,
            "voice_incomplete_n": n_excluded,
            "voice_incomplete_percent_within_response": float(Fraction(100 * n_excluded, n_total)),
            "response_percent_within_voice_complete": float(Fraction(100 * n_kept, sum(observed))),
            "response_percent_within_voice_incomplete": float(Fraction(100 * n_excluded, sum(excluded))),
            "weighting": "Unweighted counts and percentages",
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = calculate(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "voice_item_response_counts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "PASS", "response_categories": len(rows),
                      "total_eligible": 6988, "voice_complete": 6548,
                      "respondent_records_read": 0}))

if __name__ == "__main__":
    main()
