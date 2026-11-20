"""Independent arithmetic and exhaustive small-sample checks of allocation bounds."""
import csv
from fractions import Fraction
from itertools import product
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from missing_voice_bounds import bounds, compute, read_counts


class MissingVoiceBoundsTests(unittest.TestCase):
    def test_source_partition(self):
        counts, a, b = read_counts(ROOT / 'results')
        self.assertEqual(sum(n for y, n in counts.values()), 6548)
        self.assertEqual(sum(y for y, n in counts.values()), 3663)
        self.assertEqual((a, b), (210, 230))
        self.assertEqual(sum(n for y, n in counts.values()) + a + b, 6988)

    def test_saved_results_and_exact_fractions(self):
        actual = compute(ROOT / 'results')
        with (ROOT / 'results/missing_voice_allocation_bounds.csv').open(newline='') as f:
            saved = list(csv.DictReader(f))
        self.assertEqual(len(actual), 7)
        for row, stored in zip(actual, saved):
            self.assertEqual(row['estimand'], stored['estimand'])
            for field in ('observed', 'lower', 'upper'):
                self.assertAlmostEqual(row[field], float(stored[field]), places=11)
        both = actual[3]
        self.assertAlmostEqual(both['lower'], float(100 * Fraction(1301,3545)), places=11)
        self.assertAlmostEqual(both['upper'], float(100 * Fraction(1531,3565)), places=11)
        # Independent exact-rational calculation, not the implementation's helper.
        expected = [
            (Fraction(420,1039)-Fraction(647,1332), Fraction(630,1019)-Fraction(437,1352)),
            (Fraction(772,1512)-Fraction(647,1332), Fraction(982,1492)-Fraction(437,1352)),
            (Fraction(2034,3565)-Fraction(647,1332), Fraction(2244,3545)-Fraction(437,1352))
        ]
        for row, (lo, hi) in zip(actual[4:], expected):
            self.assertAlmostEqual(row['lower'], float(100*lo), places=11)
            self.assertAlmostEqual(row['upper'], float(100*hi), places=11)

    def test_exhaustive_toy_allocations(self):
        # Assign every missing yes and no to focal, baseline or another group.
        y, n, y0, n0, a, b = 2, 5, 1, 4, 2, 2
        shares, differences = [], []
        for yes_assignment in product(range(3), repeat=a):
            for other_assignment in product(range(3), repeat=b):
                ya, yb = yes_assignment.count(0), yes_assignment.count(1)
                na, nb = other_assignment.count(0), other_assignment.count(1)
                p = Fraction(y+ya, n+ya+na)
                p0 = Fraction(y0+yb, n0+yb+nb)
                shares.append(p); differences.append(p-p0)
        lo, hi = bounds(y,n,a,b)
        self.assertAlmostEqual(lo, float(min(shares)), places=14)
        self.assertAlmostEqual(hi, float(max(shares)), places=14)
        self.assertEqual(min(differences), Fraction(y,n+b)-Fraction(y0+a,n0+a))
        self.assertEqual(max(differences), Fraction(y+a,n+a)-Fraction(y0,n0+b))

if __name__ == '__main__':
    unittest.main()
