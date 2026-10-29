"""Additional rival account: specialised safety representation, offline."""
from pathlib import Path
import sys,json,io,zipfile
import argparse
ap=argparse.ArgumentParser()
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--source-zip',type=Path,required=True)
args=ap.parse_args();O=args.out.resolve();P=O/'private'
import pandas as pd,numpy as np
from scipy import stats
source=args.source_zip
with zipfile.ZipFile(source) as z:payload=z.read('UKDA-9511-stata/stata/stata13/ewcs24_dataset_ukda_v2.dta')
with pd.io.stata.StataReader(io.BytesIO(payload),convert_categoricals=False) as r:r.variable_labels();r.value_labels();d=r.read()
eu=d.country_code.isin('AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE'.split());base=eu&d.employee_selfdeclared.eq(1)&d[['tech_task_alloc','tech_hours','tech_monitor_perf']].isin([1,2]).any(axis=1)&d.aut_procedure_decision.isin([1,2,3])&d.trade_union.isin([1,2])&d.meeting.isin([1,2])
results=[]
for model,mask in [('A_with_OSH',base),('PAIRED_with_OSH',base&d.training_employer.isin([1,2]))]:
 f=d.loc[mask];name='A_procedure_yes' if model.startswith('A_') else 'PAIRED_training_minus_procedure';a=np.load(P/(name+'_verification.npz'));y=a['y'];w=a['w'];idx=a['pi'];h=a['psu_strata'];cols=list(a['columns'])
 assert len(y)==len(f)
 assert np.array_equal((f.aut_procedure_decision==1).to_numpy().astype(float),y) if name=='A_procedure_yes' else np.array_equal((f.training_employer.eq(1).astype(float)-f.aut_procedure_decision.eq(1).astype(float)).to_numpy(),y)
 # No-delegate is reference; unknown is retained separately.
 X=np.column_stack([a['X'],f.osh_delegate.eq(1).astype(float),~f.osh_delegate.isin([1,2])]);cols+=['OSH_present','OSH_unknown']
 B=np.linalg.inv(X.T@(w[:,None]*X));b=B@(X.T@(w*y));influence=((w*(y-X@b))[:,None]*X)@B;u=np.zeros((len(h),X.shape[1]));np.add.at(u,idx,influence);pieces=[];single=0
 for st in np.unique(h):
  v=u[h==st];m=len(v)
  if m==1:single+=1;continue
  v=v-v.mean(axis=0);pieces.append(v.T@v*m/(m-1))
 V=sum(pieces)*(1+single/len(pieces));df=len(h)-len(np.unique(h));c=stats.t.ppf(.975,df)
 for label,vec in [('representation_only - neither',{'representation_only':1}),('meetings_only - neither',{'meetings_only':1}),('both - meetings_only',{'both':1,'meetings_only':-1}),('additive_interaction',{'both':1,'meetings_only':-1,'representation_only':-1})]:
  q=np.zeros(len(b))
  for term,val in vec.items():q[next(i for i,t in enumerate(cols) if t.endswith('[T.'+term+']'))]=val
  est=q@b;se=np.sqrt(q@V@q);results.append({'model':model,'contrast':label,'estimate_pp':100*est,'lo95_pp':100*(est-c*se),'hi95_pp':100*(est+c*se),'se_pp':100*se,'p':float(2*stats.t.sf(abs(est/se),df)),'n':len(y)})
t=pd.DataFrame(results);t.to_csv(O/'tables/T9_OSH_sensitivity.csv',index=False,float_format='%.8f');print(t.to_string(index=False))
