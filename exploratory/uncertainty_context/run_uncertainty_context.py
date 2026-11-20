#!/usr/bin/env python3
"""One exploratory, same-person DK specificity contrast. Aggregate output only.

Run: python run_uncertainty_context.py --data /authorized/ewcs24_dataset_ukda_v2.dta --out results
Requires Python 3.10+, numpy, pandas and scipy. No network calls. No microdata output.
The script does not execute any historical outcome model.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import platform
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import linalg, stats

SOURCE_SHA = 'd5cb93c4f7f4ad148ad6f1628abe061b29252516b2c8eb98f05a05ccc9d2e49c'
SPEC_SHA = '741c93fa22e0e0255a75b1131fb68c57e82880a5cd73edd77591607e85c7c4b3'
SPEC_DOCUMENTED_DATE = '2026-11-01'
EU = 'AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE'.split()
TECH = ['tech_task_alloc', 'tech_hours', 'tech_monitor_perf']
ITEMS = ['trade_union', 'osh_delegate', 'work_accom_periods', 'work_accomodations_menopause']
REJECTED = ['wp_conflicts', 'wp_emptrustman']
CONTROLS = ['country_code', 'age_group', 'gender', 'education', 'occupational_group',
            'industry_group', 'size_group', 'contract_group', 'parttime']
OUTCOMES = ITEMS + ['representative_DK_mean', 'accommodation_DK_mean', 'representative_minus_accommodation']
PAIR_LABELS = ['Q57_DK_minus_No', 'Q57_DK_minus_Yes']
# Four item outcomes map to two domain means and their paired difference.
TRANSFORM = np.column_stack([np.eye(4), [.5,.5,0,0], [0,0,.5,.5], [.5,.5,-.5,-.5]])


def jwrite(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=lambda x: x.item())+'\n', encoding='utf-8')


def source_read(path: Path):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA:
        raise ValueError('Input is not the required authenticated V2 file.')
    with pd.io.stata.StataReader(path, convert_categoricals=False) as reader:
        varlabels = reader.variable_labels()
        labels = reader.value_labels()
        mapping = dict(zip(reader._varlist, reader._lbllist))
        data = reader.read()
    if data.shape != (36644, 394):
        raise ValueError(f'Unexpected source dimensions: {data.shape}')
    wanted = ['aut_procedure_decision'] + ITEMS + REJECTED
    # Stata release 118 stores 129-byte characteristic names. Recover full text.
    pos = raw.index(b'<characteristics>') + len(b'<characteristics>')
    notes = {}
    while raw[pos:pos+4] == b'<ch>':
        size = struct.unpack_from('<I', raw, pos+4)[0]
        record = raw[pos+8:pos+8+size]
        var = record[:129].split(b'\0')[0].decode('utf-8')
        key = record[129:258].split(b'\0')[0].decode('utf-8')
        if var in wanted and key == 'note1':
            notes[var] = record[258:].rstrip(b'\0').decode('utf-8')
        pos += 8+size+5
    metadata = {v: {'variable_label': varlabels[v], 'full_question_note': notes.get(v),
                       'value_labels': {str(int(k)): val for k,val in labels.get(mapping[v],{}).items()}}
                for v in wanted}
    return data, metadata


class SurveyDesign:
    """Country-nested, fixed-weight ultimate-cluster approximation."""
    def __init__(self, full: pd.DataFrame):
        self.full = full.reset_index(drop=True).copy()
        self.full['_stratum'] = self.full.country_code.astype(str)+'|'+self.full.stratumid.astype(str)
        self.full['_psu'] = self.full.country_code.astype(str)+'|'+self.full.samplingpointid.astype(str)
        if self.full.groupby('_psu')._stratum.nunique().max() != 1:
            raise ValueError('A PSU crosses strata.')
        self.frame = self.full[['_psu','_stratum']].drop_duplicates().reset_index(drop=True)
        lookup = dict(zip(self.frame._psu, self.frame.index))
        self.full['_pi'] = self.full._psu.map(lookup).astype(int)
        self.groups = [g.index.to_numpy() for _,g in self.frame.groupby('_stratum',sort=True)]
        self.df = sum(len(g)-1 for g in self.groups)
        self.singletons = sum(len(g)==1 for g in self.groups)

    def covariance(self, influence: np.ndarray, pi: np.ndarray) -> np.ndarray:
        influence = np.asarray(influence, float)
        if influence.ndim == 1:
            influence = influence[:,None]
        totals = np.zeros((len(self.frame),influence.shape[1]))
        np.add.at(totals, pi, influence)
        v = np.zeros((influence.shape[1],influence.shape[1])); valid = 0
        for ids in self.groups:
            m = len(ids)
            if m > 1:
                centered = totals[ids] - totals[ids].mean(axis=0)
                v += m/(m-1)*(centered.T@centered)
                valid += 1
        if valid == 0:
            raise ValueError('No nonsingleton stratum.')
        v *= 1+self.singletons/valid
        return (v+v.T)/2

    def info(self) -> dict:
        return {'EU27_full_design_n': len(self.full), 'PSUs': len(self.frame),
                'strata': len(self.groups), 'singleton_strata': self.singletons,
                'df': self.df, 'singleton_rule': 'average_nonsingleton_contribution',
                'outside_domain_scores': 0, 'released_calibration_weights_held_fixed': True,
                'finite_population_correction': False}


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d['employee'] = d.employee_selfdeclared.eq(1) | d.paid_salary.eq(1)
    d['eligible'] = d[TECH].isin([1,2]).any(axis=1) & d.aut_procedure_decision.isin([1,2,3])
    d['age_group'] = pd.cut(d.age,[15,34,54,74],labels=['16-34','35-54','55-74'],include_lowest=True).astype(str)
    d['gender'] = d.sex_id.map({1:'Man',2:'Woman',3:'Another'}).fillna('Unknown')
    d['education'] = np.select([d.isced.between(0,2),d.isced.between(3,4),d.isced.between(5,8)],['Low','Middle','High'],default='Unknown')
    d['occupational_group'] = d.occupation.astype(str)
    d['industry_group'] = d.sector.astype(str)
    d['size_group'] = d.wp_size.map({1:'1-9',2:'1-9',3:'1-9',4:'10-49',5:'50-249',6:'50-249',7:'250+',8:'250+'}).fillna('Unknown')
    d['contract_group'] = d.empl_contract.map({1:'Permanent',2:'Fixed_agency',3:'Fixed_agency',4:'Other_training',5:'No_contract',995:'Other_training'}).fillna('Unknown')
    d['parttime'] = d.part_time.where(d.part_time.isin([1,2]),0).astype(int).astype(str)
    for v in TECH:
        d['x_'+v] = d[v].where(d[v].isin([1,2,3,4]),0).astype(int).astype(str)
    return d


def model_matrix(d: pd.DataFrame, gender_age=False):
    columns = {f'q57_{k}':d.aut_procedure_decision.eq(k).to_numpy(float) for k in (1,2,3)}
    controls = CONTROLS+['x_'+v for v in TECH]
    if gender_age:
        d = d.copy()
        d['gender_age'] = d.gender+'|'+d.age_group
        controls = [v for v in controls if v not in ('gender','age_group')] + ['gender_age']
    for name in controls:
        a = d[name].fillna('Unknown').astype(str)
        for lev in sorted(a.unique())[1:]:
            columns[name+'='+lev] = a.eq(lev).to_numpy(float)
    X = np.column_stack(list(columns.values()))
    Q = np.zeros((2,X.shape[1])); Q[0,[1,2]] = [-1,1]; Q[1,[0,2]] = [-1,1]
    return X,Q,list(columns)


def fit(d: pd.DataFrame, design: SurveyDesign, model: str, weight=None, gender_age=False, full_output=True):
    X,Q,names = model_matrix(d,gender_age)
    Y = d[ITEMS].eq(-999).to_numpy(float)
    w = d.calweight.to_numpy(float) if weight is None else np.asarray(weight,float)
    if not np.isfinite(w).all() or (w<=0).any():
        raise ValueError('Nonpositive or invalid weights.')
    w = w/w.mean()
    Z = X*np.sqrt(w[:,None])
    B,_,rank,singular = linalg.lstsq(Z, Y*np.sqrt(w[:,None]), cond=1e-12)
    if rank != X.shape[1]:
        raise ValueError(f'Rank deficiency in {model}: {rank}/{X.shape[1]}')
    bread = linalg.inv(X.T@(w[:,None]*X))
    direct_error = np.max(np.abs(B-bread@(X.T@(w[:,None]*Y))))
    if direct_error > 1e-8:
        raise AssertionError('Normal-equation cross-check failed.')
    residual = Y-X@B
    h = X@bread@Q.T
    score = (h[:,:,None]*(w[:,None,None]*residual[:,None,:])).reshape(len(d),8)
    V = design.covariance(score,d._pi.to_numpy(int))
    values = Q@B
    values_all = values@TRANSFORM
    cov_all = np.kron(np.eye(2),TRANSFORM.T)@V@np.kron(np.eye(2),TRANSFORM)
    critical = stats.t.ppf(.975,design.df)
    rows = []
    for q,contrast in enumerate(PAIR_LABELS):
        for j,outcome in enumerate(OUTCOMES):
            if not full_output and j<4:
                continue
            se = np.sqrt(max(0.,cov_all[q*7+j,q*7+j])); est = values_all[q,j]
            row = {'model':model,'contrast':contrast,'outcome':outcome,'n':len(d),
                   # Detailed Q57 margins are needed only for the main domain.
                   # Omitting sensitivity margins prevents small-cell recovery
                   # by subtracting a leave-one-country-out count from the total.
                   'n_q57_yes':int(d.aut_procedure_decision.eq(1).sum()) if model=='primary' else None,
                   'n_q57_no':int(d.aut_procedure_decision.eq(2).sum()) if model=='primary' else None,
                   'n_q57_dk':int(d.aut_procedure_decision.eq(3).sum()) if model=='primary' else None,
                   'countries':d.country_code.nunique(),'df':design.df,
                   'estimate_pp':100*est,'se_pp':100*se,
                   'lo95_pp':100*(est-critical*se),'hi95_pp':100*(est+critical*se),
                   'p_pointwise':float(2*stats.t.sf(abs(est/se),design.df)) if se else None,
                   'role':'focal_exploratory' if model=='primary' and q==0 and j==6 else 'component_or_interpretive' if model=='primary' else 'sensitivity'}
            rows.append(row)
    # A separate paired-outcome fit must produce the same estimate AND variance.
    D = Y@TRANSFORM[:,-1]
    bd,*_ = linalg.lstsq(Z,D*np.sqrt(w),cond=1e-12)
    sd = h*(w*(D-X@bd))[:,None]
    vd = design.covariance(sd,d._pi.to_numpy(int))
    vd_expected = cov_all[np.ix_([6,13],[6,13])]
    paired_b_error = float(np.max(np.abs(Q@bd-values_all[:,-1])))
    paired_v_error = float(np.max(np.abs(vd-vd_expected)))
    if paired_b_error>1e-10 or paired_v_error>1e-10:
        raise AssertionError('Paired outcome and stacked-score identity failed.')
    pi = X@B
    diagnostics = {'model':model,'n':len(d),'columns':len(names),'rank':int(rank),
                   'kish_weight_effective_n':float(w.sum()**2/(w@w)),
                   'weighted_design_condition_number':float(singular.max()/singular.min()),
                   'normal_equation_max_abs_error':float(direct_error),
                   'paired_estimate_max_abs_error':paired_b_error,
                   'paired_covariance_max_abs_error':paired_v_error,
                   'joint_covariance_min_eigenvalue':float(linalg.eigvalsh(V).min()),
                   'linear_fitted_range_by_item':{v:{'min':float(pi[:,i].min()),'max':float(pi[:,i].max()),
                          'fraction_outside_0_1':float(np.mean((pi[:,i]<0)|(pi[:,i]>1)))} for i,v in enumerate(ITEMS)}}
    if model=='primary':
        diagnostics['X_column_names'] = names
        diagnostics['joint_covariance_order'] = [c+' | '+v for c in PAIR_LABELS for v in ITEMS]
        diagnostics['joint_covariance_pp2'] = (10000*V).tolist()
        diagnostics['domain_covariance'] = {}
        for q,c in enumerate(PAIR_LABELS):
            ix=[q*7+4,q*7+5]
            v=cov_all[np.ix_(ix,ix)]*10000
            diagnostics['domain_covariance'][c] = {'var_representative_pp2':float(v[0,0]),
                'var_accommodation_pp2':float(v[1,1]),'covariance_pp2':float(v[0,1]),
                'variance_difference_pp2':float(v[0,0]+v[1,1]-2*v[0,1])}
    return rows,diagnostics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    spec=Path(__file__).with_name('SPECIFICATION.md')
    if hashlib.sha256(spec.read_bytes()).hexdigest()!=SPEC_SHA:
        raise ValueError('The accompanying documented specification has changed.')
    started=dt.datetime.now(dt.timezone.utc).isoformat()
    raw,meta=source_read(args.data)
    design=SurveyDesign(raw.loc[raw.country_code.isin(EU)])
    d=prepare(design.full)
    eligible=d.loc[d.employee & d.eligible].copy()
    if len(eligible)!=6992:
        raise AssertionError('The required employee/Q57 denominator was not reproduced.')
    valid=eligible[ITEMS[:2]].isin([1,2,-999]).all(axis=1) & eligible[ITEMS[2:]].isin([1,2,3,-999]).all(axis=1)
    common=eligible.loc[valid].copy()
    if len(common)!=6674:
        raise AssertionError('The archived common-sample denominator was not reproduced.')
    # Source checks occur before statistical estimation and before export.
    for v in REJECTED:
        if raw[v].eq(-999).any() or raw[v].eq(-998).any():
            raise AssertionError('The rejected-item source audit changed.')
    support=[]
    for state in (1,2,3):
        s=common.loc[common.aut_procedure_decision.eq(state)]
        for v in ITEMS:
            n=int(s[v].eq(-999).sum()); support.extend([n,len(s)-n])
    if min(support)<10:
        raise ValueError('Explicit DK cells are too small for this contrast.')
    records=[]
    def diag(section,domain,variable,statistic,value=None,text='',q57='all'):
        records.append({'section':section,'domain':domain,'q57':q57,'variable':variable,
                        'statistic':statistic,'value':value,'text':text})
    for state in (1,2,3):
        e=eligible.loc[eligible.aut_procedure_decision.eq(state)]
        s=common.loc[common.aut_procedure_decision.eq(state)]
        diag('denominator','eligible','Q57','n',len(e),q57=state)
        diag('denominator','common','Q57','n',len(s),q57=state)
        diag('completion','eligible','four_selected_items','complete_weighted_pct',float(np.average(valid.loc[e.index],weights=e.calweight)*100),q57=state)
        for v in ITEMS:
            indicator=s[v].eq(-999)
            diag('prevalence','common',v,'explicit_DK_n',int(indicator.sum()),q57=state)
            diag('prevalence','common',v,'explicit_DK_weighted_pct',float(np.average(indicator,weights=s.calweight)*100),q57=state)
    # Only pooled special-code frequencies are exported. Suppress small nonzero
    # counts. Do not export a complementary exhaustive substantive count.
    for v in ITEMS+REJECTED:
        for code in (-999,-998,-991):
            n=int(eligible[v].eq(code).sum())
            small=0<n<10
            diag('code_audit','eligible',v,str(code)+'_n',None if small else n,'suppressed_small_nonzero' if small else 'observed_numeric_code')
        n=int(eligible[v].isna().sum())
        diag('code_audit','eligible',v,'system_missing_n',None if 0<n<10 else n,'suppressed_small_nonzero' if 0<n<10 else 'reason_not_identifiable')
    # Explicitly record Q79's full-release zero without equating it to true DK.
    for v in REJECTED:
        diag('rejected_comparator','all_36644',v,'retained_DK_n',int(raw[v].eq(-999).sum()),'label_present_but_explicit_outcome_not_observable')
    rows,baseinfo=fit(common,design,'primary')
    focal=next(r for r in rows if r['role']=='focal_exploratory')
    informative=(abs(focal['estimate_pp'])>=3 and (focal['lo95_pp']>0 or focal['hi95_pp']<0))
    fitinfo=[baseinfo]
    if informative:
        specs=[('Q6_only',common.loc[common.employee_selfdeclared.eq(1)],None,False),
               ('equal_country',common,common.calweight/common.groupby('country_code').calweight.transform('sum'),False),
               ('gender_by_age',common,None,True)]
        for label,s,w,g in specs:
            rr,ff=fit(s,design,label,w,g,full_output=False);rows+=rr;fitinfo.append(ff)
        # One country-composition scheme, all countries, no favorable subset.
        for country in sorted(common.country_code.unique()):
            s=common.loc[common.country_code.ne(country)]
            rr,ff=fit(s,design,'omit_'+country,full_output=False)
            rows += [r for r in rr if r['contrast']==PAIR_LABELS[0]]
            fitinfo.append(ff)
    pd.DataFrame(rows).to_csv(args.out/'02_CONTRASTS.csv',index=False,float_format='%.15g')
    pd.DataFrame(records).to_csv(args.out/'03_DIAGNOSTICS.csv',index=False,float_format='%.15g')
    result={'source':{'sha256':SOURCE_SHA,'rows':len(raw),'variables':len(raw.columns),
                       'dataset_doi':'10.5255/UKDA-SN-9511-2','source_outer_archive_verified':False},
            'execution':{'started_utc':started,'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                         'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                         'scipy':scipy.__version__,'historical_models_executed':False},
            'specification_record':{'documented_date':SPEC_DOCUMENTED_DATE,'specification_sha256':SPEC_SHA,
                      'stage':'post-analysis documentation of archived code','retrospective_preregistration_claim':False},
            'metadata':meta,'design':design.info(),'eligible_n':len(eligible),'common_n':len(common),
            'disclosure':{'threshold':10,'minimum_selected_DK_or_nonDK_cell':min(support),
                          'selected_cells_pass':True,'microdata_or_identifiers_exported':False,
                          'small_nonzero_code_counts_suppressed':True,
                          'sensitivity_q57_margins_omitted_to_prevent_small_cell_differencing':True,
                          'external_disclosure_approval_claim':False},
            'focal_result':focal,'conditional_sensitivity_rule_triggered':informative,
            'stop_reason':None if informative else 'Focal difference is imprecise or below the archived 3-point operational threshold; no country, weight or subgroup search.',
            'fits':fitinfo,
            'output_sha256':{n:hashlib.sha256((args.out/n).read_bytes()).hexdigest() for n in ['02_CONTRASTS.csv','03_DIAGNOSTICS.csv']}}
    jwrite(args.out/'primary_execution.json',result)
    print(json.dumps({'focal':focal,'sensitivity_trigger':informative,'models_fitted':len(fitinfo)},indent=2,default=lambda x:x.item()))

if __name__=='__main__':
    main()
