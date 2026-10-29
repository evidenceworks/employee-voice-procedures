"""Independent parser, QR solution and cluster-variance verification; offline."""
from pathlib import Path
import sys,json,csv,math,io,zipfile,hashlib
import argparse
ap=argparse.ArgumentParser()
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--source-zip',type=Path,required=True)
args=ap.parse_args();O=args.out.resolve();P=O/'private'
import numpy as np,pandas as pd,pyreadstat
from scipy.linalg import lstsq
checks={}
source=args.source_zip
with zipfile.ZipFile(source) as z:
 assert z.testzip() is None
 data=z.read('UKDA-9511-stata/stata/stata13/ewcs24_dataset_ukda_v2.dta')
 target=P/'independent_reader.dta';target.write_bytes(data)
cols=['country_code','employee_selfdeclared','trade_union','meeting','tech_task_alloc','tech_hours','tech_monitor_perf','aut_procedure_decision','calweight','design_weight','gweight','training_employer','training_asked','wt_arrangements','wt_arrangements_change','able_hour_off','change_employment']
d,meta=pyreadstat.read_dta(str(target),usecols=cols,encoding='latin1')
with pd.io.stata.StataReader(io.BytesIO(data),convert_categoricals=False) as r:
 r.variable_labels();r.value_labels();d2=r.read()
for c in cols:
 if c!='country_code':assert np.allclose(pd.to_numeric(d[c],errors='raise').to_numpy(dtype=float),pd.to_numeric(d2[c],errors='raise').to_numpy(dtype=float),equal_nan=True),c
 else:assert d[c].equals(d2[c].astype(object)) or d[c].astype(str).equals(d2[c].astype(str))
checks['independent_parser_numeric_columns']=len(cols)-1
checks['independent_parser']='PASS'
eu=set('AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE'.split())
records=d.to_dict('records')
agg={g:[0,0,0,0.,0.,0.] for g in ['neither','representation_only','meetings_only','both']}
for r in records:
 if r['country_code'] not in eu or r['employee_selfdeclared']!=1:continue
 if not any(r[x] in [1,2] for x in ['tech_task_alloc','tech_hours','tech_monitor_perf']):continue
 if r['aut_procedure_decision'] not in [1,2,3] or r['trade_union'] not in [1,2] or r['meeting'] not in [1,2]:continue
 g={ (2,2):'neither',(1,2):'representation_only',(2,1):'meetings_only',(1,1):'both'}[(r['trade_union'],r['meeting'])]
 k=int(r['aut_procedure_decision'])-1;agg[g][k]+=1;agg[g][k+3]+=r['calweight']
table=list(csv.DictReader((O/'tables/T1_Procedure_by_voice.csv').open(encoding='utf8')))
for r in table:
 k=int(r['outcome'])-1;g=agg[r['group']]
 assert int(r['cell_n'])==g[k]
 assert abs(float(r['weighted_pct'])-100*g[k+3]/sum(g[3:]))<.000051
checks['independent_standard_library_aggregate']='PASS'
qr_errors=[];var_errors=[]
for n in ['A_procedure_yes','PAIRED_training_minus_procedure']:
 a=np.load(P/(n+'_verification.npz'))
 X=a['X'];y=a['y'];w=a['w'];idx=a['pi'];h=a['psu_strata']
 beta=lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),lapack_driver='gelsy')[0]
 assert np.linalg.matrix_rank(X)==X.shape[1]
 qr_errors.append(float(np.max(np.abs(beta-a['b']))));assert qr_errors[-1]<1e-8
 # Independently aggregate transformed influence functions using dictionary grouping.
 Ainv=np.linalg.inv(X.T@(w[:,None]*X))
 influence=((w*(y-X@beta))[:,None]*X)@Ainv
 sums=np.zeros((len(h),X.shape[1]));np.add.at(sums,idx,influence)
 V=np.zeros_like(a['V']);pieces=[];single=0
 for st in np.unique(h):
  u=sums[h==st];m=len(u)
  if m==1:single+=1;continue
  centered=u-u.mean(axis=0)
  pieces.append((centered.T@centered)*m/(m-1))
 V=sum(pieces)*(1+single/len(pieces))
 var_errors.append(float(np.max(np.abs(V-a['V']))));assert var_errors[-1]<1e-8
checks['qr_solution']='PASS';checks['max_coefficient_disagreement']=max(qr_errors)
checks['independent_influence_variance']='PASS';checks['max_covariance_disagreement']=max(var_errors)
checks['full_rank_main_and_paired']='PASS'
for file in (O/'tables').glob('T[1-6]_*.csv'):
 t=pd.read_csv(file)
 numeric=pd.to_numeric(t.cell_n,errors='coerce')
 if numeric.notna().all():
  assert numeric.min()>=10
  for _,g in t.groupby('group'):
   assert g.cell_n.astype(int).sum()==int(g.group_n.iloc[0])
   assert abs(g.weighted_pct.astype(float).sum()-100)<.001
 else:assert t[['cell_n','group_n','weighted_pct']].eq('SUPPRESSED').all().all()
checks['whole_block_disclosure_and_margins']='PASS'
qa=json.loads((P/'analysis_summary.json').read_text())
assert qa['alg_routing_outside_answers']==qa['schedule_outside_answers']==qa['training_outside_answers']==0
assert qa['N_primary']==sum(sum(x[:3]) for x in agg.values())==6548
assert qa['N_alg_routed_employee']==7038
checks['routing_and_denominators']='PASS'
checks['no_microdata_in_release']='Checked by separate package allowlist; private matrices excluded'
checks['status']='PASS'
(O/'VALIDATION.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
print(json.dumps(checks,indent=2))
