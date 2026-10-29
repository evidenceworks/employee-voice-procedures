"""Tests catch aggregate corruption and preserve the inferential contract."""
import unittest,sys,tempfile,shutil,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from check_aggregates import check

class AggregateTests(unittest.TestCase):
 def test_released_results(self):self.assertEqual(check(ROOT/'results')['status'],'PASS')
 def test_altered_interval_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for p in (ROOT/'results').glob('*.csv'):shutil.copyfile(p,root/p.name)
   p=root/'T7_Model_contrasts.csv'
   with p.open(newline='',encoding='utf-8') as f:r=list(csv.DictReader(f))
   r[0]['hi95_pp']=str(float(r[0]['hi95_pp'])+1)
   with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=r[0]);w.writeheader();w.writerows(r)
   with self.assertRaises(AssertionError):check(root)

if __name__=='__main__':unittest.main()
