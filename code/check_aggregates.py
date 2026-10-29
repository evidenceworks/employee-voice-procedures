"""Independent checks of released aggregates; no licensed records required."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from scipy.stats import t

def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def check(root):
 tables={p.name:read(p) for p in root.glob('T[0-9]_*.csv')}
 assert len(tables)==10 and sum(map(len,tables.values()))==350
 infer=sum((rows for name,rows in tables.items() if name.startswith(('T7_','T8_','T9_'))),[])
 assert len(infer)==277
 err=0
 for r in infer:
  b,se,lo,hi,p=map(float,(r[k] for k in ['estimate_pp','se_pp','lo95_pp','hi95_pp','p']))
  df=26 if r['model']=='A_country_cluster' else 2455
  q=t.ppf(.975,df)
  e=max(abs(lo-(b-q*se)),abs(hi-(b+q*se)))
  assert e<1e-7,(r['model'],r['contrast'],e)
  assert abs(p-2*t.sf(abs(b/se),df))<1e-7
  err=max(err,e)
 t7=tables['T7_Model_contrasts.csv'];lk={(r['model'],r['contrast']):r for r in t7}
 cs=['representation_only - neither','meetings_only - neither','both - meetings_only']
 raw=np.array([2*t.sf(abs(float(lk['A_procedure_yes',c]['estimate_pp'])/float(lk['A_procedure_yes',c]['se_pp'])),2455) for c in cs])
 ix=np.argsort(raw);adj=np.empty(3);adj[ix]=np.minimum(1,np.maximum.accumulate(raw[ix]*np.arange(3,0,-1)))
 assert max(abs(adj[i]-float(lk['A_procedure_yes',c]['holm_p_primary'])) for i,c in enumerate(cs))<1e-7
 assert sum(bool(r.get('holm_p_primary')) for r in infer)==3
 contrast_names=[r['contrast'] for r in t7 if r['model']=='A_procedure_yes']
 for c in contrast_names:
  assert abs(sum(float(lk['A_procedure_'+o,c]['estimate_pp']) for o in ['yes','no','unknown']))<3e-8
 base=read(root/'BASELINE_Q57.csv');eigs={}
 for o in ['yes','no','unknown']:
  rs={c:lk['A_procedure_'+o,c] for c in contrast_names}
  v={c:float(r['se_pp'])**2 for c,r in rs.items()}
  vr,vm=v[cs[0]],v[cs[1]]
  vb=v['both - meetings_only']+v['both - representation_only']+vr+vm-v['meetings_only - representation_only']-v['additive_interaction']
  crm=(vr+vm-v['meetings_only - representation_only'])/2
  crb=(vr+vb-v['both - representation_only'])/2
  cmb=(vm+vb-v['both - meetings_only'])/2
  cov=np.array([[vr,crm,crb],[crm,vm,cmb],[crb,cmb,vb]])
  ev=np.linalg.eigvalsh(cov);assert min(ev)>0;eigs[o]=ev.tolist()
  b=float(rs['meetings_only - neither']['estimate_pp'])+float(rs['both - meetings_only']['estimate_pp'])
  row=next(r for r in base if r['outcome']=='procedure_'+o and r['contrast']=='both - neither')
  assert abs(float(row['estimate_pp'])-b)<3e-8
  assert abs(float(row['se_pp'])-np.sqrt(vb))<1e-7
  for c,w in [('representation_only - neither',[1,0,0]),('meetings_only - neither',[0,1,0]),('both - meetings_only',[0,-1,1]),('both - representation_only',[-1,0,1]),('meetings_only - representation_only',[-1,1,0]),('additive_interaction',[-1,-1,1])]:
   assert abs(np.array(w)@cov@np.array(w)-v[c])<1e-7
 t1=tables['T1_Procedure_by_voice.csv']
 # Whole profile cells, including uncertainty, must remain visible.
 assert len(t1)==12
 for g in ['neither','representation_only','meetings_only','both']:
  rr=[r for r in t1 if r['group']==g]
  assert len(rr)==3 and {r['outcome'] for r in rr}=={'1','2','3'}
  assert abs(sum(float(r['weighted_pct']) for r in rr)-100)<.0002
 counts=[int(float(r['cell_n'])) for r in t1]
 assert sum(counts)==6548 and min(counts)==118
 assert sum(int(r['source_employee_n']) for r in tables['T0_Country_coverage.csv'])==23995
 for r in t7:
  if 'LOCO_' in r['model']:assert not r['n']
 return {'status':'PASS','empirical_table_rows':350,'inferential_records':277,'maximum_CI_reconstruction_difference_pp':err,'response_partition':'PASS','three_contrast_Holm':'PASS','derived_covariance_eigenvalues':eigs,'profile_counts':'PASS','country_denominator':'PASS','withheld_omission_denominators':'PASS'}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--receipt',type=Path);a=p.parse_args();r=check(a.results)
 if a.receipt:a.receipt.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(r,indent=2))
