"""EWCS 2024 V2: executed, offline, disclosure-controlled aggregate analyses.
Input: licensed UKDS SN9511 archive (never redistributed).
Python 3.12; run from any directory; optional --source-zip and --out.
"""
from pathlib import Path
import os,sys,json,io,zipfile,hashlib,argparse,platform,importlib.metadata as im
HERE=Path(__file__).resolve().parent
import numpy as np,pandas as pd
from scipy import stats
import patsy
import statsmodels.api as sm

EU='AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE'.split()
VOICE=['neither','representation_only','meetings_only','both']
MINCELL=10
ap=argparse.ArgumentParser()
ap.add_argument('--source-zip',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();OUT=a.out.resolve();P=OUT/'private'
for sub in ['private','tables','figures']:(OUT/sub).mkdir(parents=True,exist_ok=True)
source=a.source_zip
assert hashlib.sha256(source.read_bytes()).hexdigest()=='33634a7d6c45b781edb05d329494bf925f78b561f8d0ef89e25cf7ee699dacf0', 'Unexpected licensed source archive'
with zipfile.ZipFile(source) as z:
 assert z.testzip() is None
 member='UKDA-9511-stata/stata/stata13/ewcs24_dataset_ukda_v2.dta'
 payload=z.read(member)
with pd.io.stata.StataReader(io.BytesIO(payload),convert_categoricals=False) as reader:
 labels=reader.variable_labels();vl=reader.value_labels();raw=reader.read()
assert raw.shape==(36644,394)
assert not raw.uniquerespid.duplicated().any()
assert np.array_equal(raw.EU27.eq(1),raw.country_code.isin(EU))
assert (np.isfinite(raw.calweight)&raw.calweight.gt(0)).all()
assert raw[['samplingpointid','stratumid']].notna().all().all()
# Stable identifiers, no respondent-level output.
full=raw.loc[raw.country_code.isin(EU)].copy().reset_index(drop=True)
full['country']=full.country_code.astype(str)
full['strata']=full.country+'_'+full.stratumid.astype(str)
full['psu']=full.country+'_'+full.samplingpointid.astype(str)
assert full.groupby('psu').strata.nunique().max()==1
psu_frame=full[['psu','strata','country']].drop_duplicates().reset_index(drop=True)
psumap=pd.Series(np.arange(len(psu_frame)),index=psu_frame.psu)
full['pi']=full.psu.map(psumap)
stratum_groups=[g.index.to_numpy() for _,g in psu_frame.groupby('strata',sort=True)]
DF=sum(len(g)-1 for g in stratum_groups)
singletons=sum(len(g)==1 for g in stratum_groups)
crit=stats.t.ppf(.975,DF)

def covariance(scores,idx,method='average',cluster_country=False):
 """WR ultimate-cluster Taylor sandwich; full design incl. zero domain scores."""
 sums=np.zeros((len(psu_frame),scores.shape[1]));np.add.at(sums,idx,scores)
 if cluster_country:
  country=pd.DataFrame(sums).groupby(psu_frame.country).sum().to_numpy()
  return (country.T@country)*len(country)/(len(country)-1)
 blocks=[];lonely=[]
 for ids in stratum_groups:
  u=sums[ids];m=len(ids)
  if m>1:
   u=u-u.mean(axis=0);blocks.append(m/(m-1)*(u.T@u))
  else:lonely.append(ids[0])
 meat=sum(blocks)
 if lonely and method=='average':meat=meat+len(lonely)*sum(blocks)/len(blocks)
 elif lonely and method=='center':
  grand=sums.mean(axis=0)
  for k in lonely:
   u=sums[k]-grand;meat=meat+np.outer(u,u)
 elif method not in ['average','center','certainty']:raise ValueError(method)
 return meat

def cat(x):
 return x.astype('string').fillna('Unknown').astype(str)

d=full.copy()
d['employee']=d.employee_selfdeclared.eq(1)
d['employee_extended']=d.employee|d.paid_salary.eq(1)
d['repr']=d.trade_union.map({1:1.,2:0.})
d['meet']=d.meeting.map({1:1.,2:0.})
d['voice']=pd.Categorical(np.select([(d.repr==0)&(d.meet==0),(d.repr==1)&(d.meet==0),(d.repr==0)&(d.meet==1),(d.repr==1)&(d.meet==1)],VOICE,default='Unknown'),categories=VOICE+['Unknown'])
d['ageband']=pd.cut(d.age,[15,24,34,44,54,64,74],labels=['16-24','25-34','35-44','45-54','55-64','65-74'],include_lowest=True).astype(str)
d['gender']=d.sex_id.map({1:'Man',2:'Woman',3:'Another'}).fillna('Unknown')
d['education']=pd.Series(np.select([d.isced.between(0,2),d.isced.between(3,4),d.isced.between(5,8)],['Low','Middle','High'],default='Unknown'),index=d.index)
d['occ']=d.ISCO_1.map({0:'Managers_professionals',1:'Managers_professionals',2:'Managers_professionals',3:'Technicians_clerks',4:'Technicians_clerks',5:'Service_sales',6:'Manual_skilled',7:'Manual_skilled',8:'Manual_skilled',9:'Elementary'}).fillna('Unknown')
d['industry']=d.NACE0_lbl.astype(str)
d['size']=d.wp_size.map({1:'1-9',2:'1-9',3:'1-9',4:'10-49',5:'50-249',6:'50-249',7:'250+',8:'250+'}).fillna('Unknown')
d['contract']=d.empl_contract.map({1:'Permanent',2:'Fixed_agency',3:'Fixed_agency',4:'Other_training',5:'No_contract',995:'Other_training'}).fillna('Unknown')
d['parttime']=cat(d.part_time.where(d.part_time.isin([1,2])))
for c in ['tech_task_alloc','tech_hours','tech_monitor_perf','wp_conflicts','wp_emptrustman','consulted']:
 d[c+'_c']=cat(d[c].where(d[c].gt(0)))
d['alg']=d[['tech_task_alloc','tech_hours','tech_monitor_perf']].isin([1,2]).any(axis=1)
d['alg_small']=d[['tech_task_alloc','tech_hours','tech_monitor_perf']].isin([1,2,3]).any(axis=1)
d['alg_large']=d[['tech_task_alloc','tech_hours','tech_monitor_perf']].eq(1).any(axis=1)
d['procvalid']=d.aut_procedure_decision.isin([1,2,3])
for name,v in [('procedure_yes',1),('procedure_no',2),('procedure_unknown',3)]:
 d[name]=d.aut_procedure_decision.eq(v).astype(float).where(d.procvalid)
d['notice']=d.wt_arrangements_change.map({1:'Stable',2:'Short_notice',3:'Short_notice',4:'Longer_notice',5:'Longer_notice'})
d['off_difficult']=d.able_hour_off.isin([3,4]).astype(float).where(d.able_hour_off.isin([1,2,3,4]))
d['private_disruption']=d.shortnotice.isin([1,2,3]).astype(float).where(d.shortnotice.isin([1,2,3,4,5]))
d['trainstate']=np.select([d.training_employer.eq(1),d.training_employer.eq(2)&d.training_asked.eq(1),d.training_employer.eq(2)&d.training_asked.eq(2)],['Trained','Untrained_requested','Untrained_no_request'],default='Unknown')
d['training']=d.training_employer.eq(1).astype(float).where(d.training_employer.isin([1,2]))
d['untrained_requested']=pd.Series(d.trainstate=='Untrained_requested',index=d.index).astype(float).where(d.trainstate!='Unknown')
d['voice_demand']=d.improve_voice.astype(float).where(d.improve_DK.eq(0)&d.improve_Refused.eq(0))
d['frequent_consult']=d.consulted.isin([1,2]).astype(float).where(d.consulted.isin([1,2,3,4,5]))
d['influence']=d.decision_influence.isin([1,2]).astype(float).where(d.decision_influence.isin([1,2,3,4,5]))
d['bullying']=d.asb_bully.eq(1).astype(float).where(d.asb_bully.isin([1,2]))
d['osh']=d.osh_delegate.map({1:1.,2:0.})
d['downsized']=d.change_employment.map({1:0.,2:0.,3:0.,4:1.,5:1.})
d['product_weight']=d.design_weight*d.gweight

CONTROL='C(country)+C(ageband)+C(gender)+C(education)+C(occ)+C(industry)+C(size)+C(contract)+C(parttime)'
VTERM="C(voice, Treatment(reference='neither'))"
TECH='C(tech_task_alloc_c)+C(tech_hours_c)+C(tech_monitor_perf_c)'
MAIN=VTERM+'+'+CONTROL+'+'+TECH
results=[];models={};coef_private={};release_audit=[]

def fit(name,y,mask,rhs=MAIN,weight='calweight',lonely='average',logit=False,vcountry=False,calibration=False):
 mask=mask&d[y].notna()
 f=d.loc[mask].copy()
 if isinstance(f.voice.dtype,pd.CategoricalDtype):f['voice']=f.voice.cat.remove_unused_categories()
 if weight=='equal_country':
  f['w']=f.calweight/f.groupby('country').calweight.transform('sum')
 elif weight=='none':f['w']=1.
 elif weight=='trim99':f['w']=f.calweight.clip(upper=f.groupby('country').calweight.transform(lambda s:s.quantile(.99)))
 else:f['w']=f[weight]
 f['w']=f.w/f.w.mean()
 Xdf=patsy.dmatrix('1+'+rhs,f,return_type='dataframe');X=np.asarray(Xdf);Y=f[y].to_numpy();w=f.w.to_numpy()
 if logit:
  model=sm.GLM(Y,X,family=sm.families.Binomial(),freq_weights=w).fit(maxiter=100)
  beta=np.asarray(model.params);mu=model.predict(X);bread=np.linalg.pinv(X.T@((w*mu*(1-mu))[:,None]*X))
 else:
  bread=np.linalg.pinv(X.T@(w[:,None]*X));beta=bread@(X.T@(w*Y));mu=X@beta
 scores=(w*(Y-mu))[:,None]*X
 if calibration:
  # Approximate GREG residualization on the supplied national calibration margins.
  # Bounded calibration and national solver details are not reconstructed exactly.
  unweighted_scores=np.zeros((len(d),X.shape[1]));unweighted_scores[f.index]=X*(Y-mu)[:,None]
  residuals=unweighted_scores.copy()
  for country,sub in d.groupby('country',sort=True):
   Z=np.asarray(patsy.dmatrix('1+C(agesex)+C(occupation)+C(sector)+C(region)',sub,return_type='dataframe'))
   qw=sub.calweight.to_numpy()/f.calweight.mean()
   inv=np.linalg.pinv(Z.T@(qw[:,None]*Z));gamma=inv@(Z.T@(qw[:,None]*unweighted_scores[sub.index]))
   residuals[sub.index]-=Z@gamma
  weights=d.calweight.to_numpy()/f.calweight.mean()
  meat=covariance(residuals*weights[:,None],d.pi.to_numpy(),lonely,vcountry)
 else:meat=covariance(scores,f.pi.to_numpy(),lonely,vcountry)
 vcov=bread@meat@bread
 info={'n':len(f),'y':y,'rhs':rhs,'weight':weight,'lonely':lonely,'logit':logit,'rank':int(np.linalg.matrix_rank(X)),'p':X.shape[1],'n_countries':f.country.nunique(),'kish_n':float(w.sum()**2/(w@w)),'range_mu':[float(mu.min()),float(mu.max())]}
 models[name]=(f,Xdf,beta,vcov,info)
 coef_private[name]={'info':info,'coefficients':dict(zip(Xdf.columns,beta.tolist())),'se':dict(zip(Xdf.columns,np.sqrt(np.maximum(np.diag(vcov),0)).tolist()))}
 print('FIT',name,json.dumps(info),flush=True)
 return name

def contrast(name,label,vector):
 f,Xdf,beta,vcov,info=models[name]
 q=np.zeros(len(beta))
 for term,val in vector.items():
  term=next(c for c in Xdf.columns if c.endswith('[T.'+term+']')) if term in VOICE[1:] else term
  if term in Xdf.columns:q[Xdf.columns.get_loc(term)]=val
  else:raise ValueError((term,Xdf.columns.tolist()))
 est=q@beta;se=np.sqrt(max(0,q@vcov@q));rdf=26 if label.startswith('country_cluster:') else DF
 ci=stats.t.ppf(.975,rdf)
 row={'model':name,'outcome':info['y'],'contrast':label,'estimate_pp':100*est,'se_pp':100*se,'lo95_pp':100*(est-ci*se),'hi95_pp':100*(est+ci*se),'p':2*stats.t.sf(abs(est/se),rdf) if se>0 else np.nan,'n':info['n']}
 results.append(row);return row

def voicecontrasts(name):
 contrast(name,'representation_only - neither',{'representation_only':1})
 contrast(name,'meetings_only - neither',{'meetings_only':1})
 contrast(name,'both - meetings_only',{'both':1,'meetings_only':-1})
 contrast(name,'both - representation_only',{'both':1,'representation_only':-1})
 contrast(name,'meetings_only - representation_only',{'meetings_only':1,'representation_only':-1})
 contrast(name,'additive_interaction',{'both':1,'representation_only':-1,'meetings_only':-1})

def safe_distribution(name,mask,group,outcome,levels):
 f=d.loc[mask&d[outcome].isin(levels)].copy()
 blocks=[];allcounts=[]
 for g,sub in f.groupby(group,observed=True,dropna=False):
  counts=[int(sub[outcome].eq(l).sum()) for l in levels];allcounts+=counts
  blocks.append((g,sub,counts))
 safe=bool(blocks) and all(c>=MINCELL for c in allcounts)
 rows=[]
 for g,sub,counts in blocks:
  for level,n in zip(levels,counts):
   rows.append({'group':str(g),'outcome':str(level),'cell_n':n if safe else 'SUPPRESSED','group_n':len(sub) if safe else 'SUPPRESSED','weighted_pct':round(100*sub.loc[sub[outcome].eq(level),'calweight'].sum()/sub.calweight.sum(),4) if safe else 'SUPPRESSED'})
 release_audit.append({'table':name,'blocks':len(blocks),'minimum_component':min(allcounts) if safe else 'SUPPRESSED','entire_table_released':safe})
 pd.DataFrame(rows).to_csv(OUT/'tables'/(''+name+'.csv'),index=False)

emp=d.employee;known=d.voice!='Unknown';alg=emp&known&d.alg&d.procvalid
safe_distribution('T1_Procedure_by_voice',alg,'voice','aut_procedure_decision',[1,2,3])
safe_distribution('T2_Training_by_voice',emp&known,'voice','trainstate',['Trained','Untrained_requested','Untrained_no_request'])
safe_distribution('T3_Absence_by_notice',emp&d.wt_arrangements.isin([1,2])&d.notice.notna(),'notice','off_difficult',[0,1])
safe_distribution('T4_Voice_demand',emp&known,'voice','voice_demand',[0,1])
safe_distribution('T5_Bullying_by_OSH',emp&d.osh.notna(),'osh','bullying',[0,1])
safe_distribution('T6_Influence_by_downsizing',emp&d.downsized.notna(),'downsized','influence',[0,1])

for y in ['procedure_yes','procedure_no','procedure_unknown']:
 name=fit('A_'+y,y,alg);voicecontrasts(name)
for y in ['training','untrained_requested','voice_demand']:
 name=fit('pilot_'+y,y,emp&known,rhs=VTERM+'+'+CONTROL);voicecontrasts(name)
fit('pilot_time','off_difficult',emp&known&d.wt_arrangements.isin([1,2])&d.notice.notna(),rhs="C(notice, Treatment(reference='Stable'))*"+VTERM+'+'+CONTROL)
contrast('pilot_time','Short_notice - Stable',{"C(notice, Treatment(reference='Stable'))[T.Short_notice]":1})
for v in VOICE[1:]:
 term="C(notice, Treatment(reference='Stable'))[T.Short_notice]:"+VTERM+'[T.'+v+']'
 contrast('pilot_time','Short_notice interaction '+v,{term:1})
fit('pilot_osh','bullying',emp&d.osh.notna(),rhs='osh+'+CONTROL);contrast('pilot_osh','OSH delegate association',{'osh':1})
fit('pilot_downsizing','influence',emp&known&d.downsized.notna(),rhs='downsized*'+VTERM+'+'+CONTROL);contrast('pilot_downsizing','Downsized association',{'downsized':1})

# Full analysis of candidate A with all adverse sensitivities retained.
for name,mask,rhs,weight,lonely in [
 ('A_unadjusted',alg,VTERM,'calweight','average'),
 ('A_country',alg,VTERM+'+C(country)','calweight','average'),
 ('A_climate',alg,MAIN+'+C(wp_conflicts_c)+C(wp_emptrustman_c)','calweight','average'),
 ('A_consulted',alg,MAIN+'+C(consulted_c)','calweight','average'),
 ('A_large',alg&d.alg_large,MAIN,'calweight','average'),
 ('A_yesno',alg&d.aut_procedure_decision.isin([1,2]),MAIN,'calweight','average'),
 ('A_extended',d.employee_extended&known&d.alg&d.procvalid,MAIN,'calweight','average'),
 ('A_equal_country',alg,MAIN,'equal_country','average'),
 ('A_designweight',alg,MAIN,'design_weight','average'),
 ('A_unweighted',alg,MAIN,'none','average'),
 ('A_trim99',alg,MAIN,'trim99','average'),
 ('A_productweight',alg,MAIN,'product_weight','average'),
 ('A_lonely_center',alg,MAIN,'calweight','center'),
 ('A_lonely_certainty',alg,MAIN,'calweight','certainty'),
]:
 fit(name,'procedure_yes',mask,rhs=rhs,weight=weight,lonely=lonely);voicecontrasts(name)

# Same-person comparison selected after the six pilots; explicitly exploratory.
d['training_minus_procedure']=d.training-d.procedure_yes
paired=alg&d.training.notna()
for y in ['training','procedure_yes','training_minus_procedure']:
 fit('PAIRED_'+y,y,paired);voicecontrasts('PAIRED_'+y)
for label,rhs,wt in [('climate',MAIN+'+C(wp_conflicts_c)+C(wp_emptrustman_c)','calweight'),('equal_country',MAIN,'equal_country'),('designweight',MAIN,'design_weight'),('unweighted',MAIN,'none')]:
 name='PAIRED_'+label;fit(name,'training_minus_procedure',paired,rhs=rhs,weight=wt);voicecontrasts(name)
for c in EU:
 name='PAIRED_LOCO_'+c;fit(name,'training_minus_procedure',paired&d.country.ne(c))
 contrast(name,'both - meetings_only',{'both':1,'meetings_only':-1})

fit('A_country_cluster','procedure_yes',alg,vcountry=True)
for lab,q in [('representation_only - neither',{'representation_only':1}),('meetings_only - neither',{'meetings_only':1}),('both - meetings_only',{'both':1,'meetings_only':-1})]:
 contrast('A_country_cluster','country_cluster:'+lab,q)
fit('A_calibration_residual','procedure_yes',alg,calibration=True);voicecontrasts('A_calibration_residual')
fit('PAIRED_calibration_residual','training_minus_procedure',paired,calibration=True);voicecontrasts('PAIRED_calibration_residual')

# Logistic standardized risk contrasts: fixed observed standardization distribution.
fit('A_logit','procedure_yes',alg,logit=True)
f,Xd,b,vc,info=models['A_logit'];means={};grads={}
for v in VOICE:
 cf=f.copy();cf['voice']=pd.Categorical([v]*len(f),categories=VOICE)
 Xnew=np.asarray(patsy.build_design_matrices([Xd.design_info],cf)[0]);pr=stats.logistic.cdf(Xnew@b);w=f.w.to_numpy();w=w/w.sum()
 means[v]=float(w@pr);grads[v]=(w*pr*(1-pr))@Xnew
for u,v in [('representation_only','neither'),('meetings_only','neither'),('both','meetings_only'),('both','representation_only')]:
 q=grads[u]-grads[v];est=means[u]-means[v];se=np.sqrt(max(0,q@vc@q))
 results.append({'model':'A_logit','outcome':'procedure_yes','contrast':u+' - '+v,'estimate_pp':100*est,'se_pp':100*se,'lo95_pp':100*(est-crit*se),'hi95_pp':100*(est+crit*se),'p':2*stats.t.sf(abs(est/se),DF),'n':len(f)})

# Leave-one-country-out stability, deliberately no country-specific outcome counts.
loco=[]
for c in sorted(EU):
 name=fit('LOCO_'+c,'procedure_yes',alg&d.country.ne(c))
 for label,vec in [('meetings_only - neither',{'meetings_only':1}),('both - meetings_only',{'both':1,'meetings_only':-1})]:
  row=contrast(name,label,vec);loco.append(row)

# Primary family Holm adjustment.
res=pd.DataFrame(results);res['holm_p_primary']=np.nan
ix=res.index[(res.model=='A_procedure_yes')&res.contrast.isin(['representation_only - neither','meetings_only - neither','both - meetings_only'])]
order=ix[np.argsort(res.loc[ix,'p'].to_numpy())];adjusted=np.maximum.accumulate(res.loc[order,'p'].to_numpy()*np.arange(3,0,-1));res.loc[order,'holm_p_primary']=np.minimum(adjusted,1)
# LOCO cannot reveal rare country exclusions through near-identical denominators: omit n.
res.loc[res.model.str.contains('LOCO_'),'n']=np.nan
res.to_csv(OUT/'tables'/'T7_Model_contrasts.csv',index=False,float_format='%.8f')
pd.DataFrame([dict(model=k,**v[4]) for k,v in models.items() if not k.startswith('LOCO_')]).to_json(P/'model_diagnostics.json',orient='records',indent=2)
(P/'coefficients.json').write_text(json.dumps(coef_private,indent=2),encoding='utf8')
(P/'logit_margins.json').write_text(json.dumps(means,indent=2),encoding='utf8')
for name in ['A_procedure_yes','PAIRED_training_minus_procedure']:
 f,X,b,V,info=models[name]
 np.savez_compressed(P/(name+'_verification.npz'),X=np.asarray(X),y=f[info['y']].to_numpy(),w=f.w.to_numpy(),pi=f.pi.to_numpy(),psu_strata=pd.factorize(psu_frame.strata)[0],columns=np.asarray(X.columns,dtype=str),b=b,V=V)

# Safe coverage: employee count only; no near-equal valid denominators.
coverage=[{'country':c,'included':True,'source_employee_n':int((d.country.eq(c)&emp).sum()),'reason':'EU27 harmonised face-to-face V2; all included before comparative modelling'} for c in sorted(EU)]
assert min(r['source_employee_n'] for r in coverage)>=10
pd.DataFrame(coverage).to_csv(OUT/'tables'/'T0_Country_coverage.csv',index=False)
keyvars=['country_code','employee_selfdeclared','paid_salary','calweight','design_weight','gweight','stratumid','samplingpointid','trade_union','meeting','tech_task_alloc','tech_hours','tech_monitor_perf','aut_procedure_decision','training_employer','training_asked','wt_arrangements','wt_arrangements_change','able_hour_off','shortnotice','consulted','decision_influence','wp_conflicts','wp_emptrustman','improve_voice','osh_delegate','asb_bully','change_employment']
cb=next((P/'source').rglob('*codebook*ukds_v2.xlsx'),None)
if cb:
 book=pd.read_excel(cb,sheet_name='Variables');book=book.set_index('variable')
else:book=None
maps=[]
def clean_label(x):
 if isinstance(x,str) and ('â' in x or 'Ã' in x):
  try:return x.encode('latin1').decode('utf8')
  except (UnicodeError,ValueError):pass
 return x
for v in keyvars:
 row=book.loc[v] if book is not None and v in book.index else {}
 x=raw[v];missing=int(x.isna().sum());small=0<missing<10
 maps.append({'variable':v,'label':row.get('Variable description',labels.get(v,'')),'question':row.get('Questionnaire item number',''),'routing':row.get('Routing',''),'value_labels':json.dumps({str(k):clean_label(val) for k,val in vl.get(v,{}).items()},ensure_ascii=False),'source_nonmissing_n':len(x)-missing if not small else 'SUPPRESSED','source_missing_n':missing if not small else 'SUPPRESSED','negative_special_n':int(x.lt(0).sum()) if pd.api.types.is_numeric_dtype(x) and int(x.lt(0).sum())>=10 else ('0' if not pd.api.types.is_numeric_dtype(x) or not x.lt(0).any() else 'SUPPRESSED')})
pd.DataFrame(maps).to_csv(OUT/'02_VARIABLE_MAP.csv',index=False)
qa={'N_source':len(raw),'N_variables':raw.shape[1],'N_eu27':len(d),'N_eu27_employee':int(emp.sum()),'N_eu27_employee_extended':int(d.employee_extended.sum()),'N_alg_routed_employee':int((emp&d.alg).sum()),'N_alg_answered_employee':int((emp&d.alg&d.procvalid).sum()),'N_primary':int(alg.sum()),'N_procedure_yesno_primary':int((alg&d.aut_procedure_decision.isin([1,2])).sum()),'N_smallonly_employee':int((emp&d.alg_small&~d.alg).sum()),'alg_routing_outside_answers':int((~d.alg&d.procvalid).sum()),'schedule_outside_answers':int((~d.wt_arrangements.isin([1,2])&d.wt_arrangements_change.notna()).sum()),'training_outside_answers':int((~d.training_employer.eq(2)&d.training_asked.notna()).sum()),'source_dta_sha256':hashlib.sha256(payload).hexdigest(),'source_zip_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'psus_eu27':len(psu_frame),'strata_eu27':len(stratum_groups),'singletons':singletons,'survey_df':DF,'positive_weight_n':int((raw.calweight>0).sum()),'weight_ratio_error_max':float(np.max(np.abs(raw.calweight/(raw.design_weight*raw.gweight)-1))),'primary_nonresponse_n':int((emp&d.alg&~d.procvalid).sum()),'primary_unknown_voice_n':int((emp&d.alg&d.procvalid&~known).sum())}
qa['outcome_counts_primary']={str(k):int((alg&d.aut_procedure_decision.eq(k)).sum()) for k in [1,2,3]}
qa['primary_weighted_outcomes']={str(k):float(d.loc[alg&d.aut_procedure_decision.eq(k),'calweight'].sum()/d.loc[alg,'calweight'].sum()) for k in [1,2,3]}
qa['versions']={m:im.version(m) for m in ['pandas','numpy','scipy','statsmodels','patsy','openpyxl']};qa['python']=platform.python_version()
(P/'analysis_summary.json').write_text(json.dumps(qa,indent=2),encoding='utf8')
pd.DataFrame(release_audit).to_csv(OUT/'DISCLOSURE_CHECK.csv',index=False)
print('SUMMARY',json.dumps(qa,indent=2),flush=True)
print(res.loc[res.model.isin(['A_procedure_yes','A_procedure_no','A_procedure_unknown','A_climate','A_logit','pilot_time','pilot_training','pilot_untrained_requested','pilot_voice_demand','pilot_osh','pilot_downsizing'])].to_string(index=False),flush=True)
