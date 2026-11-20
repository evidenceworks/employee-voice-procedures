#!/usr/bin/env python3
"""Independent reconstruction of the new test, not a rerun of its estimator.

Run: python 05_verify_independent.py --data authorized.dta --results results --out independent.json
Uses a separate design-matrix implementation, statsmodels WLS and a different
stratum covariance formula. It does not import or call 04_run_decisive.py.
An additional Stata-118 binary-storage audit checks the selected item columns
without using pandas' observation reader. No respondent records are exported.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from scipy.stats import t

SHA = 'd5cb93c4f7f4ad148ad6f1628abe061b29252516b2c8eb98f05a05ccc9d2e49c'
EU = set('AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE'.split())
ITEMS = ['trade_union','osh_delegate','work_accom_periods','work_accomodations_menopause']
TECH = ['tech_task_alloc','tech_hours','tech_monitor_perf']
OUTCOMES = ITEMS + ['representative_DK_mean','accommodation_DK_mean','representative_minus_accommodation']


def check_raw_storage(blob: bytes, df: pd.DataFrame) -> dict:
    def section(tag):
        start=blob.index(('<'+tag+'>').encode())+len(tag)+2
        end=blob.index(('</'+tag+'>').encode(),start)
        return blob[start:end]
    if section('release')!=b'118' or section('byteorder')!=b'LSF':
        raise ValueError('This independent binary check supports Stata 118 little-endian only.')
    k=struct.unpack('<H',section('K'))[0]; n=struct.unpack('<Q',section('N'))[0]
    types=np.frombuffer(section('variable_types'),dtype='<u2')
    namebytes=section('varnames')
    names=[namebytes[i*129:(i+1)*129].split(b'\0')[0].decode('utf-8') for i in range(k)]
    widths={65526:8,65527:4,65528:4,65529:2,65530:1,32768:8}
    sizes=[widths[int(x)] if int(x) in widths else int(x) for x in types]
    stride=sum(sizes); offsets=np.cumsum([0]+sizes[:-1]); rawdata=section('data')
    if len(rawdata)!=n*stride:
        raise ValueError('Independent record-width check failed.')
    dtype={65526:'<f8',65527:'<f4',65528:'<i4',65529:'<i2',65530:'i1'}
    missing={65526:float.fromhex('0x1.0000000000000p+1023'),
             65527:float.fromhex('0x1.0000000000000p+127'),
             65528:2147483621,65529:32741,65530:101}
    report={}
    for var in ['aut_procedure_decision']+ITEMS+['wp_conflicts','wp_emptrustman']:
        j=names.index(var); typ=int(types[j])
        vals=np.ndarray((n,),dtype=dtype[typ],buffer=rawdata,offset=int(offsets[j]),strides=(stride,)).astype(float)
        mask=vals>=missing[typ]
        raw_missing_kinds=int(len(np.unique(vals[mask])))
        all_system=bool((vals[mask]==missing[typ]).all())
        vals[mask]=np.nan
        ref=df[var].to_numpy(float)
        equal=bool(np.array_equal(np.isnan(vals),np.isnan(ref)) and np.allclose(vals,ref,equal_nan=True,atol=0,rtol=0))
        if not equal:
            raise AssertionError('Raw binary and pandas disagree: '+var)
        dk_code=3 if var=='aut_procedure_decision' else -999
        report[var]={'observations_checked':n,'exact_numeric_and_missing_match':equal,
                     'observed_raw_missing_kinds':raw_missing_kinds,'all_missing_are_system_dot':all_system,
                     'explicit_DK_code':dk_code,'full_source_explicit_DK_count':int(np.sum(vals==dk_code))}
    return {'stata_release':118,'rows':n,'columns':k,'row_bytes':stride,'fields':report}


def controls(d: pd.DataFrame, interaction: bool) -> np.ndarray:
    # Independently encoded with pandas get_dummies rather than the main loop.
    c=pd.DataFrame(index=d.index)
    c['country']=d.country_code.astype(str)
    c['age']=pd.cut(d.age,[15,34,54,74],labels=['16-34','35-54','55-74'],include_lowest=True).astype(str)
    c['gender']=d.sex_id.map({1:'Man',2:'Woman',3:'Another'}).fillna('Unknown')
    c['education']='Unknown'
    c.loc[d.isced.between(0,2),'education']='Low'
    c.loc[d.isced.between(3,4),'education']='Middle'
    c.loc[d.isced.between(5,8),'education']='High'
    c['occupation']=d.occupation.astype(str);c['sector']=d.sector.astype(str)
    c['size']=d.wp_size.map({1:'1-9',2:'1-9',3:'1-9',4:'10-49',5:'50-249',6:'50-249',7:'250+',8:'250+'}).fillna('Unknown')
    c['contract']=d.empl_contract.map({1:'Permanent',2:'Fixed_agency',3:'Fixed_agency',4:'Other_training',5:'No_contract',995:'Other_training'}).fillna('Unknown')
    c['parttime']=d.part_time.where(d.part_time.isin([1,2]),0).astype(int).astype(str)
    for v in TECH:
        c[v]=d[v].where(d[v].isin([1,2,3,4]),0).astype(int).astype(str)
    if interaction:
        c['gender_age']=c.gender+'|'+c.age
        c=c.drop(columns=['gender','age'])
    dummy=pd.get_dummies(c,drop_first=True,dtype=float)
    q=np.stack([(d.aut_procedure_decision==k).to_numpy(float) for k in (1,2,3)],axis=1)
    return np.concatenate([q,dummy.to_numpy(float)],axis=1)


def separate_covariance(scores: np.ndarray, sample: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    # Sum of outer products minus outer product of sums, with full PSU reindexing.
    cols=list(range(scores.shape[1]))
    z=pd.DataFrame(scores,columns=cols)
    z['psu']=sample['_p'].to_numpy()
    totals=z.groupby('psu',sort=True)[cols].sum().reindex(frame.index,fill_value=0)
    totals['stratum']=frame['_s']
    variance=np.zeros((len(cols),len(cols)));multi=0;single=0
    for _,g in totals.groupby('stratum',sort=True):
        m=len(g)
        if m==1:
            single+=1
            continue
        a=g[cols].to_numpy(); sums=a.sum(axis=0)
        variance+=(m*(a.T@a)-np.outer(sums,sums))/(m-1)
        multi+=1
    return variance*(1+single/multi)


def reconstruct(d: pd.DataFrame, frame: pd.DataFrame, model: str, dfree: int):
    X=controls(d,model=='gender_by_age')
    w=d.calweight.to_numpy(float)
    if model=='equal_country':
        w=(d.calweight/d.groupby('country_code').calweight.transform('sum')).to_numpy(float)
    items=d[ITEMS].eq(-999).to_numpy(float)
    I=(items[:,0]+items[:,1])/2;A=(items[:,2]+items[:,3])/2
    Y=np.column_stack([items,I,A,I-A])
    Q=np.zeros((2,X.shape[1]));Q[0,1]=-1;Q[0,2]=1;Q[1,0]=-1;Q[1,2]=1
    # Multivariate WLS reuses the independently computed pseudoinverse across
    # outcomes. The paired outcome is then fitted again as a separate scalar
    # model. This avoids recomputing seven identical design decompositions.
    fit=sm.WLS(Y,X,weights=w).fit(method='pinv')
    estimates=Q@fit.params
    h=X@fit.normalized_cov_params@Q.T
    residual=Y-X@fit.params
    influence=(h[:,:,None]*(w[:,None,None]*residual[:,None,:])).reshape(len(d),14)
    paired=sm.WLS(I-A,X,weights=w).fit(method='pinv')
    paired_b_error=float(np.max(np.abs(Q@paired.params-estimates[:,6])))
    if paired_b_error>1e-10:
        raise AssertionError('Independent scalar paired-outcome identity failed.')
    V=separate_covariance(influence,d,frame)
    out={}
    for q,name in enumerate(['Q57_DK_minus_No','Q57_DK_minus_Yes']):
        for j,outcome in enumerate(OUTCOMES):
            b=100*estimates[q,j];s=100*np.sqrt(V[q*7+j,q*7+j]);h=t.ppf(.975,dfree)*s
            out[(name,outcome)]={'estimate_pp':b,'se_pp':s,'lo95_pp':b-h,'hi95_pp':b+h,
                                 'p_pointwise':2*t.sf(abs(b/s),dfree)}
    # Identity check includes covariance of the two exposure contrasts.
    L=np.zeros((2,14));L[0,[4,5]]=[1,-1];L[1,[11,12]]=[1,-1]
    error=float(np.max(np.abs(L@V@L.T-V[np.ix_([6,13],[6,13])])))
    if error>1e-10:
        raise AssertionError('Independent direct/stacked covariance identity failed.')
    scalar_h=X@paired.normalized_cov_params@Q.T
    scalar_scores=scalar_h*(w*(I-A-X@paired.params))[:,None]
    scalar_v=separate_covariance(scalar_scores,d,frame)
    scalar_v_error=float(np.max(np.abs(scalar_v-L@V@L.T)))
    if scalar_v_error>1e-10:
        raise AssertionError('Independent scalar-pair covariance check failed.')
    return out,V,error,paired_b_error,scalar_v_error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    started=dt.datetime.now(dt.timezone.utc).isoformat()
    blob=args.data.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=SHA:
        raise ValueError('Wrong source file.')
    with pd.io.stata.StataReader(args.data,convert_categoricals=False) as reader:
        reader.variable_labels();reader.value_labels();raw=reader.read()
    print('Independent source read complete.',flush=True)
    storage=check_raw_storage(blob,raw)
    print('Independent binary-storage check complete.',flush=True)
    full=raw.loc[raw.country_code.isin(EU)].copy()
    full['_p']=full.country_code.astype(str)+'|'+full.samplingpointid.astype(str)
    full['_s']=full.country_code.astype(str)+'|'+full.stratumid.astype(str)
    frame=full[['_p','_s']].drop_duplicates().set_index('_p').sort_index()
    dfree=len(frame)-frame._s.nunique()
    eligible=(full.employee_selfdeclared.eq(1)|full.paid_salary.eq(1)) & full[TECH].isin([1,2]).any(axis=1) & full.aut_procedure_decision.isin([1,2,3])
    observed=full[ITEMS[:2]].isin([1,2,-999]).all(axis=1) & full[ITEMS[2:]].isin([1,2,3,-999]).all(axis=1)
    sample=full.loc[eligible & observed].copy()
    if int(eligible.sum())!=6992 or len(sample)!=6674:
        raise AssertionError('Independent sample reconstruction failed.')
    target=pd.read_csv(args.results/'02_CONTRASTS.csv')
    metadata=json.loads((args.results/'primary_execution.json').read_text())
    differences=[];model_reports=[];baseV=None
    for model in target.model.unique():
        print('Verifying '+model,flush=True)
        ss=sample
        if model=='Q6_only':
            ss=sample.loc[sample.employee_selfdeclared.eq(1)]
        if model.startswith('omit_'):
            ss=sample.loc[sample.country_code.ne(model[5:])]
        actual,V,identity,paired_b_error,scalar_v_error=reconstruct(ss,frame,model,dfree)
        if model=='primary':baseV=V
        rows=target.loc[target.model.eq(model)]
        errors=[]
        for _,r in rows.iterrows():
            if int(r.n)!=len(ss) or int(r.df)!=dfree:
                raise AssertionError('Sample size or df mismatch.')
            for field in ('estimate_pp','se_pp','lo95_pp','hi95_pp','p_pointwise'):
                e=abs(float(r[field])-actual[(r.contrast,r.outcome)][field])
                errors.append(e);differences.append(e)
        model_reports.append({'model':model,'rows_checked':len(rows),'numeric_cells_checked':len(errors),
                               'maximum_absolute_difference':max(errors),
                               'direct_pair_stacked_covariance_max_error':identity,
                               'scalar_paired_estimate_max_error':paired_b_error,
                               'scalar_paired_covariance_max_error':scalar_v_error})
    indices=[0,1,2,3,7,8,9,10]
    itemV=baseV[np.ix_(indices,indices)]*10000
    covdiff=float(np.max(np.abs(itemV-np.asarray(metadata['fits'][0]['joint_covariance_pp2']))))
    if max(differences)>1e-7 or covdiff>1e-7:
        raise AssertionError('Independent numeric tolerance failed.')
    # Aggregate formulas also checked without model refitting.
    p0=target.loc[target.model.eq('primary')].set_index(['contrast','outcome'])
    arithmetic=[]
    for c in ['Q57_DK_minus_No','Q57_DK_minus_Yes']:
        a=p0.loc[(c,'representative_DK_mean'),'estimate_pp']
        b=p0.loc[(c,'accommodation_DK_mean'),'estimate_pp']
        delta=p0.loc[(c,'representative_minus_accommodation'),'estimate_pp']
        arithmetic.append(abs(a-b-delta))
    report={'status':'PASS','started_utc':started,'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'independence_scope':'Separate code and numerical implementation in the same environment, not an external-person audit or a separate software-language replication.',
            'primary_script_imported_or_called':False,'statsmodels_version':statsmodels.__version__,
            'source_sha256':SHA,'raw_binary_check':storage,
            'eligible_n':int(eligible.sum()),'common_n':len(sample),'design_df':dfree,
            'models_verified':len(model_reports),'aggregate_rows_verified':len(target),
            'numeric_cells_verified':len(differences),'maximum_absolute_numeric_difference':max(differences),
            'joint_covariance_cells_verified':64,'joint_covariance_max_abs_difference_pp2':covdiff,
            'aggregate_difference_identity_max_error_pp':max(arithmetic),
            'tolerance':1e-7,'models':model_reports,
            'microdata_or_identifiers_exported':False}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['status','models_verified','aggregate_rows_verified','numeric_cells_verified','maximum_absolute_numeric_difference','joint_covariance_max_abs_difference_pp2']},indent=2))

if __name__=='__main__':
    main()
