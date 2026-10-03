"""Synthetic moment algebra and offline artifact boundaries; no actual data read."""
import copy
import csv
import json

import numpy as np
import pytest

from experiments.cvs_frontfilter_analysis import coefficient_association as module


def factor_grid():
    contrasts=[]
    for size in (6,5,3):
        x=np.arange(size,dtype=np.float64);x-=x.mean();contrasts.append(x/np.sqrt(np.mean(x*x)))
    t=contrasts[0][:,None,None];r=contrasts[1][None,:,None];d=contrasts[2][None,None,:]
    return [np.broadcast_to(x,(6,5,3)) for x in (t,r,d,t*r,t*d,r*d,t*r*d)]


@pytest.mark.parametrize('index',range(7))
def test_each_pure_main_or_interaction_recovers_its_population_energy(index):
    grid=np.full((6,5,3,8),.03)
    grid[...,0]+=.02*factor_grid()[index]
    result=module.decompose_coefficients(grid,np.zeros((6,5,3)))
    for j,factor in enumerate(module.FACTORS[:-1]):
        assert result['components'][factor]==pytest.approx(.0004 if index==j else 0.,abs=1e-18)
    assert result['components']['within_cell']==0.
    assert result['total_trace_variance']==pytest.approx(.0004)
    assert result['fractions'][module.FACTORS[index]]==pytest.approx(1.)
    assert result['global_mean']==pytest.approx([.03]*8)


def test_all_effects_add_orthogonally_even_in_the_same_coefficient_component():
    amplitudes=np.arange(1,8)*.001
    grid=np.zeros((6,5,3,8));grid[...,0]=sum(a*x for a,x in zip(amplitudes,factor_grid()))
    within=np.full((6,5,3),.0003)
    result=module.decompose_coefficients(grid,within)
    for factor,a in zip(module.FACTORS,amplitudes):assert result['components'][factor]==pytest.approx(a*a)
    assert result['components']['within_cell']==pytest.approx(.0003)
    expected=float(np.square(amplitudes).sum())+.0003
    assert result['total_trace_variance']==pytest.approx(expected)
    assert result['reconstructed_total_trace_variance']==pytest.approx(expected)
    assert sum(result['fractions'].values())==pytest.approx(1.)
    assert result['cell_mean_reconstruction_max_abs_error']<1e-16
    # Translation must not be mistaken for factor variation.
    translated=module.decompose_coefficients(grid+.1,within)
    assert translated['components']==pytest.approx(result['components'])


@pytest.mark.parametrize('within',(0.,3e-12,1e-10))
def test_constant_or_cancellation_scale_variance_keeps_raw_values_and_null_ratios(within):
    result=module.decompose_coefficients(np.full((6,5,3,8),.12345),np.full((6,5,3),within))
    assert result['total_trace_variance']==pytest.approx(within,abs=1e-25)
    assert result['components']['within_cell']==pytest.approx(within,abs=1e-25)
    assert all(result['components'][k]==0. for k in module.FACTORS[:-1])
    assert all(value is None for value in result['fractions'].values())
    assert result['raw_variances_preserved'] is True


@pytest.mark.parametrize('case',('means_shape','within_shape','nan_mean','inf_within','negative_within','bool_mean','overflow'))
def test_invalid_math_input_is_rejected(case):
    x=np.zeros((6,5,3,8));w=np.zeros((6,5,3))
    if case=='means_shape':x=x[:5]
    elif case=='within_shape':w=w[:5]
    elif case=='nan_mean':x[0,0,0,0]=np.nan
    elif case=='inf_within':w[0,0,0]=np.inf
    elif case=='negative_within':w[0,0,0]=-1e-15
    elif case=='bool_mean':x=x.astype(bool)
    elif case=='overflow':x[0,0,0,0]=1e308
    with pytest.raises(ValueError):module.decompose_coefficients(x,w)


@pytest.fixture
def rows():
    result=[];grid=factor_grid()[0]
    for variant in module.VARIANTS:
        for s,seed in enumerate(module.SEEDS):
            for ti,t in enumerate(module.TX):
                for ri,r in enumerate(module.RX):
                    for di,d in enumerate(module.DAYS):
                        mean=[.05]*8
                        if variant==module.VARIANTS[1]:mean[0]+=float(.001*(s+1)*grid[ti,ri,di])
                        result.append(dict(variant=variant,model_seed=seed,tx=t,receiver=r,day=d,count=300,
                            coefficient_mean=mean,coefficient_trace_variance=1e-12 if variant==module.VARIANTS[0] else .0001,
                            relative_input_change_mean=.01,relative_input_change_max=.02,
                            coefficient_l1_mean=.4,coefficient_l1_max=.5,kernel_l1_mean=.3,kernel_l1_max=.4,
                            input_norm_ratio_eligible_count=300,input_norm_ratio_min=.99,input_norm_ratio_max=1.01))
    return result


def test_all_eight_models_use_four_seed_variance_sd_without_mutating_inputs(rows):
    rows.reverse();before=copy.deepcopy(rows)
    result=module.analyze_cells(rows)
    assert rows==before and len(result['models'])==8 and len(result['four_seed_summary'])==18
    tx=next(r for r in result['four_seed_summary'] if r['variant']==module.VARIANTS[1] and r['factor']=='TX')
    energies=np.square(np.arange(1,5)*.001)
    assert tx['trace_variance_mean']==pytest.approx(energies.mean())
    assert tx['trace_variance_seed_sd']==pytest.approx(energies.std(ddof=1))
    assert tx['fraction_mean']==pytest.approx(np.mean(energies/(energies+.0001)))
    assert tx['fraction_defined_seeds']==4
    assert all(r['fractions']['TX'] is None for r in result['models'] if r['variant']==module.VARIANTS[0])
    assert result['model_selection'] is False and result['target_access'] is False
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('case',('missing','duplicate','variant','seed','count','TX','RX','day','nonfinite',
    'negative','dimension','bool_metadata','bool_coefficient','static_variance','static_change','extra_field','norm_null'))
def test_complete_source_cell_contract_rejects_invalid_schema(rows,case):
    row=rows[0]
    if case=='missing':rows.pop()
    elif case=='duplicate':rows[-1]=copy.deepcopy(rows[-2])
    elif case=='variant':row['variant']='unregistered'
    elif case=='seed':row['model_seed']=123
    elif case=='count':row['count']=299
    elif case=='TX':row['tx']=6
    elif case=='RX':row['receiver']=2
    elif case=='day':row['day']=4
    elif case=='nonfinite':row['coefficient_mean'][0]=float('nan')
    elif case=='negative':row['coefficient_trace_variance']=-1e-12
    elif case=='dimension':row['coefficient_mean'].pop()
    elif case=='bool_metadata':row['tx']=False
    elif case=='bool_coefficient':row['coefficient_mean'][0]=True
    elif case=='static_variance':row['coefficient_trace_variance']=1e-7
    elif case=='static_change':row['coefficient_mean'][0]+=.01
    elif case=='extra_field':row['unregistered']=1
    elif case=='norm_null':row['input_norm_ratio_min']=None
    with pytest.raises(ValueError):module.analyze_cells(rows)


def save_json(path,value):
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')


def source_artifact(tmp_path,rows):
    evidence=tmp_path/'automation_reports/CV-SincNet'/module.RUN/'evidence';evidence.mkdir(parents=True)
    selection=dict(status='SOURCE_SELECTION_FROZEN',candidate_universe=['neural_residual_shallow','response_anchor_mean',*module.VARIANTS],
        selected_variant='response_anchor_mean',new_candidate_selected=False,target_access=False,target_score_used=False)
    completion=dict(status='VERIFIED',new_rows=8,control_rows=8,new_epochs=1600,new_steps=80000,full_stdout_scanned=True,
        step_epoch_csv_stdout_reconciled=True,full_source_records_reverified=True,source_rule_recomputed=True,
        target_access=False,source_selection=selection)
    analysis=dict(status='VERIFIED',models=8,source_epochs=1600,control_epochs=1600,source_frontfilter_cells=720,
        target_results_read=False,selection=selection)
    models=[]
    for variant in module.VARIANTS:
        for seed in module.SEEDS:
            cells=[{k:v for k,v in r.items() if k not in ('variant','model_seed')} for r in rows if r['variant']==variant and r['model_seed']==seed]
            models.append(dict(resolved=dict(variant=variant,model_seed=seed),
                completion=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False),
                source_diagnostics=dict(role='V',count=27000,target_access=False,used_for_training=False,used_for_selection=False,
                    groups=[{k:c[k] for k in ('tx','receiver','day','count')} for c in cells],frontfilter_groups=cells)))
    source=dict(ready=True,run_id=module.RUN,pipeline=dict(status='SOURCE_RESEARCH_COMPLETE_BASELINE_RETAINED'),source_selection=selection,rows=models)
    for name,data in (('source_selection',selection),('source_completion_validation',completion),('source_analysis_validation',analysis),('source_research_complete',source)):
        save_json(evidence/(name+'.json'),data)
    with (evidence/'source_frontfilter_cells.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return evidence


def test_offline_complete_source_roundtrip_preserves_sources_and_control_winner(tmp_path,rows):
    evidence=source_artifact(tmp_path,rows);before={p.name:p.read_bytes() for p in evidence.iterdir()}
    result=module.run(tmp_path)
    assert result['source_selection']['selected_variant']=='response_anchor_mean'
    assert result['status']=='VERIFIED' and len(result['models'])==8
    assert all((evidence/name).read_bytes()==data for name,data in before.items())
    with (evidence/'source_coefficient_association_by_seed.csv').open(encoding='utf-8',newline='') as stream:
        output=list(csv.DictReader(stream))
    assert len(output)==8 and len(json.loads(output[0]['global_mean']))==8
    text=(evidence/'source_coefficient_association.md').read_text(encoding='utf-8')
    assert text.startswith('# Source coefficient association')
    for phrase in ('非因果','不能证明信道恢复','禁止用于选模','不除以 300','原值保留','不保证跨模型对齐'):
        assert phrase in text
    assert json.loads((evidence/'source_coefficient_association.json').read_text(encoding='utf-8'))['target_access'] is False


@pytest.mark.parametrize('case',('missing_freeze','freeze_disagrees','partial_analysis','partial_budget','live_source',
    'missing_model','wrong_input_class','changed_csv','target','wrong_run'))
def test_only_existing_terminal_consistent_source_evidence_can_write_report(tmp_path,rows,case):
    evidence=source_artifact(tmp_path,rows)
    name='source_research_complete'
    if case=='missing_freeze':(evidence/'source_selection.json').unlink()
    elif case=='changed_csv':
        path=evidence/'source_frontfilter_cells.csv';path.write_text(path.read_text(encoding='utf-8').replace('0.05','0.04',1),encoding='utf-8')
    else:
        if case in ('freeze_disagrees','partial_analysis'):name='source_analysis_validation'
        elif case=='partial_budget':name='source_completion_validation'
        data=module.read_json(evidence/(name+'.json'))
        if case=='freeze_disagrees':data['selection']['selected_variant']='neural_residual_shallow'
        elif case=='partial_analysis':data['source_frontfilter_cells']=719
        elif case=='partial_budget':data['new_steps']=79999
        elif case=='live_source':data['pipeline']['status']='SOURCE_TRAINING'
        elif case=='missing_model':data['rows'].pop()
        elif case=='wrong_input_class':data['rows'][0]['source_diagnostics']['groups'][0]['tx']=6
        elif case=='target':data['rows'][0]['completion']['target_access']=True
        elif case=='wrong_run':data['run_id']='other'
        save_json(evidence/(name+'.json'),data)
    with pytest.raises((ValueError,FileNotFoundError)):module.run(tmp_path)
    assert not list(evidence.glob('source_coefficient_association*'))
