"""Checks for covariance-derived comparisons and item-response denominators."""
import csv
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from response_state_comparisons import calculate,source_lookup
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
                self.assertAlmostEqual(float(a[field]),float(b[field]),places=9)

    def test_voice_response_counts(self):
        actual=response_counts(ROOT/'results')
        with (ROOT/'results/voice_item_response_counts.csv').open(newline='',encoding='utf-8') as stream:
            expected=list(csv.DictReader(stream))
        self.assertEqual(len(actual),3)
        for a,b in zip(actual,expected):
            for field in a:
                if isinstance(a[field],float):self.assertAlmostEqual(a[field],float(b[field]),places=11)
                else:self.assertEqual(str(a[field]),b[field])

if __name__=='__main__':unittest.main()
