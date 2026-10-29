"""Reconstruct all released empirical tables from licensed source records offline."""
from pathlib import Path
import argparse, subprocess, sys, json, hashlib, zipfile, csv, io, math, time

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--source-zip',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True,help='New, empty output directory')
 a=ap.parse_args();source=a.source_zip.resolve();out=a.out.resolve()
 if out.exists() and any(out.iterdir()):raise SystemExit('Output directory must be empty')
 out.mkdir(parents=True,exist_ok=True);private=out/'private';private.mkdir()
 root=Path(__file__).resolve().parent.parent
 sha=hashlib.sha256(source.read_bytes()).hexdigest()
 assert sha=='33634a7d6c45b781edb05d329494bf925f78b561f8d0ef89e25cf7ee699dacf0'
 with zipfile.ZipFile(source) as z:
  assert z.testzip() is None
  members=[n for n in z.namelist() if 'codebook' in n.lower() and n.endswith('ukds_v2.xlsx')]
  assert len(members)==1
  dest=private/'source';dest.mkdir();(dest/Path(members[0]).name).write_bytes(z.read(members[0]))
 receipt={'source_zip_sha256':sha,'steps':[],'comparison_tolerance':1e-6,'status':'RUNNING'}
 (out/'clean_run.json').write_text(json.dumps(receipt,indent=2))
 for name in ['analyze.py','verify_analysis.py','logistic_interaction.py','safety_representation.py']:
  t=time.monotonic()
  result=subprocess.run([sys.executable,str(root/'code'/name),'--source-zip',str(source),'--out',str(out)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
  (private/(name+'.log')).write_text(result.stdout,encoding='utf-8')
  receipt['steps'].append({'script':name,'exit_code':result.returncode,'elapsed_seconds':round(time.monotonic()-t,2)})
  if result.returncode:
   receipt['status']='FAIL';(out/'clean_run.json').write_text(json.dumps(receipt,indent=2));raise SystemExit(result.stdout[-4000:])
  print(name,'PASS',flush=True)
 comparisons=[]
 for expected in sorted((root/'results').glob('T*.csv')):
  actual=out/'tables'/expected.name
  e=list(csv.DictReader(expected.open(encoding='utf-8-sig')));b=list(csv.DictReader(actual.open(encoding='utf-8-sig')))
  assert len(e)==len(b),(expected.name,'rows')
  max_error=0.
  for i,(x,y) in enumerate(zip(e,b)):
   assert x.keys()==y.keys()
   for k in x:
    if x[k]==y[k]:continue
    try:
     xx,yy=float(x[k]),float(y[k]);err=abs(xx-yy)
     assert math.isnan(xx) and math.isnan(yy) or err<=1e-6,(expected.name,i,k,xx,yy)
     if math.isfinite(err):max_error=max(max_error,err)
    except ValueError:raise AssertionError((expected.name,i,k,x[k],y[k]))
  comparisons.append({'table':expected.name,'rows':len(e),'max_absolute_difference':max_error,'status':'PASS'})
 q=json.loads((private/'analysis_summary.json').read_text())
 expected_counts={'N_source':36644,'N_variables':394,'N_eu27':28621,'N_eu27_employee':23995,'N_alg_routed_employee':7038,'N_alg_answered_employee':6988,'N_primary':6548,'N_procedure_yesno_primary':5581,'psus_eu27':2857,'strata_eu27':402,'singletons':6,'survey_df':2455,'primary_unknown_voice_n':440}
 for k,v in expected_counts.items():assert q[k]==v,(k,q[k],v)
 receipt.update(status='PASS',tables=comparisons,total_rows=sum(x['rows'] for x in comparisons),sample_and_design=expected_counts,source_dta_sha256=q['source_dta_sha256'],weight_ratio_error_max=q['weight_ratio_error_max'],independent_checks=json.loads((out/'VALIDATION.json').read_text()))
 (out/'clean_run.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
 print('COMPLETE SOURCE-TO-OUTPUT RECONSTRUCTION PASS',flush=True)

if __name__=='__main__':main()
