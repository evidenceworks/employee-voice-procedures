"""Logistic additive interaction on the standardized probability scale, offline."""
from pathlib import Path
import sys,json
import argparse
ap=argparse.ArgumentParser()
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--source-zip',type=Path,required=True)
args=ap.parse_args();O=args.out.resolve();P=O/'private'
import numpy as np,pandas as pd
from scipy import stats
import statsmodels.api as sm
a=np.load(P/'A_procedure_yes_verification.npz');X=a['X'];y=a['y'];w=a['w'];idx=a['pi'];h=a['psu_strata'];cols=list(a['columns'])
fit=sm.GLM(y,X,family=sm.families.Binomial(),freq_weights=w).fit(maxiter=100)
b=fit.params;p=fit.predict(X);B=np.linalg.inv(X.T@((w*p*(1-p))[:,None]*X))
influence=(w*(y-p))[:,None]*X@B
u=np.zeros((len(h),X.shape[1]));np.add.at(u,idx,influence)
pieces=[];single=0
for st in np.unique(h):
 v=u[h==st];m=len(v)
 if m==1:single+=1;continue
 v=v-v.mean(axis=0);pieces.append(v.T@v*m/(m-1))
V=sum(pieces)*(1+single/len(pieces));means={};grad={}
voicecols={v:next(i for i,c in enumerate(cols) if c.endswith('[T.'+v+']')) for v in ['representation_only','meetings_only','both']}
for v in ['neither','representation_only','meetings_only','both']:
 z=X.copy();z[:,list(voicecols.values())]=0
 if v!='neither':z[:,voicecols[v]]=1
 p=stats.logistic.cdf(z@b);means[v]=float(w@p/w.sum());grad[v]=((w/w.sum())*p*(1-p))@z
q=grad['both']-grad['representation_only']-grad['meetings_only']+grad['neither']
est=means['both']-means['representation_only']-means['meetings_only']+means['neither']
se=float(np.sqrt(q@V@q));df=len(h)-len(np.unique(h));c=stats.t.ppf(.975,df)
r={'model':'A_logit_standardized_interaction','outcome':'procedure_yes','contrast':'additive_interaction','estimate_pp':100*est,'se_pp':100*se,'lo95_pp':100*(est-c*se),'hi95_pp':100*(est+c*se),'p':float(2*stats.t.sf(abs(est/se),df)),'n':len(y)}
pd.DataFrame([r]).to_csv(O/'tables/T8_Logistic_interaction.csv',index=False,float_format='%.8f')
(P/'logistic_interaction.json').write_text(json.dumps({'contrast':r,'adjusted_means':means},indent=2),encoding='utf8')
print(json.dumps({'contrast':r,'adjusted_means':means},indent=2))
