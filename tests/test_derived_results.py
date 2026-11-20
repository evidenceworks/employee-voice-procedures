"""Checks for covariance-derived comparisons and item-response denominators."""
import csv
import math
import tempfile
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from response_state_comparisons import calculate,source_lookup
from verify_public_results import compare_derived_table
from voice_item_response import calculate as response_counts

class DerivedResultsTests(unittest.TestCase):
    def test_response_comparisons(self):
        actual,_=calculate(source_lookup(results=ROOT/'results'))
        with (ROOT/'results/response_state_comparisons.csv').open(newline='',encoding='utf-8') as stream:
            expected=list(csv.DictReader(stream))
        self.assertEqual(len(actual),len(expected))
        for a,b in zip(actual,expected):
            self.assertEqual(a['configuration_contrast'],b['configuration_contrast'])
            for field in ('estimate_pp','se_pp','lo95_pp','hi95_pp','p_pointwise'):
                self.assertTrue(
                    math.isfinite(float(a[field])) and math.isclose(
                        float(a[field]), float(b[field]), rel_tol=1e-10, abs_tol=1e-8),
                    f'{a["configuration_contrast"]}: {field}')

    def test_voice_response_counts(self):
        actual=response_counts(ROOT/'results')
        with (ROOT/'results/voice_item_response_counts.csv').open(newline='',encoding='utf-8') as stream:
            expected=list(csv.DictReader(stream))
        self.assertEqual(len(actual),3)
        for a,b in zip(actual,expected):
            for field in a:
                if isinstance(a[field],float):self.assertAlmostEqual(a[field],float(b[field]),places=11)
                else:self.assertEqual(str(a[field]),b[field])


    def test_numeric_rounding_does_not_fail_reproduction(self):
        released = ROOT / 'results' / 'response_state_comparisons.csv'
        with tempfile.TemporaryDirectory() as folder:
            recalculated = Path(folder) / released.name
            with released.open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            rows[0]['p_pointwise'] = str(float(rows[0]['p_pointwise']) + 1e-11)
            with recalculated.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            compare_derived_table(released, recalculated)

    def test_material_difference_in_derived_result_is_rejected(self):
        released = ROOT / 'results' / 'response_state_comparisons.csv'
        with tempfile.TemporaryDirectory() as folder:
            recalculated = Path(folder) / released.name
            with released.open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            rows[0]['estimate_pp'] = str(float(rows[0]['estimate_pp']) + 0.01)
            with recalculated.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(AssertionError, 'estimate_pp'):
                compare_derived_table(released, recalculated)

if __name__=='__main__':unittest.main()
