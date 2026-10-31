import copy
import math
from pathlib import Path
import pytest
from experiments.pamf_phase2_practical import r42_suite_analysis as analysis
from experiments.pamf_phase2_practical.io import read,write
from experiments.pamf_phase2_practical.r42_suite_worker import desired_rows


def fixture(tmp_path,monkeypatch):
    rt=dict(run_root=str(tmp_path/'run'),run_id='suite',model_seeds=[1,2])
    jobs=[dict(key=f'j{s}',seed=s,capsule='capsules/P01/MIX01/rx',points=[[5,20]]) for s in (1,2)]
    jobs.append(dict(key='scene',seed=1,capsule='capsules/P02/LS01/rx',points=[[5,20]]))
    monkeypatch.setattr(analysis,'tasks',lambda rt:jobs)
    for job in jobs:
        folder=Path(rt['run_root'])/'tasks'/job['key'];score=folder/'scores/rx';score.mkdir(parents=True)
        _,pool,episode,rx=job['capsule'].split('/');rows=[]
        for r in desired_rows(job['points']):
            n=r['new_count'];count=6+n;classes=[f'class{i}' for i in range(count)];q=int(r['input_path'][-1]);base=.8 if job['seed']==1 else .6
            value=base+({0:0,1:.02,2:.05}[q] if r['arm']=='R42' else {0:-.08,1:-.05}[q])
            metrics=dict(count=count*55,accuracy=value,macro_accuracy=value,old_macro_accuracy=value,
                new_macro_accuracy=value if r['stage']=='C' else None,H_macro=value if r['stage']=='C' else None,
                macro_f1=value,class_floor=value-.1,class_counts=[55]*count,class_recall=[value]*count)
            rows.append(dict(r,split_id=f'split{n}',registered_classes=classes,physical_pool=pool,episode=episode,receiver=rx,
                model_seed=job['seed'],query_count=count*55,prediction='metadata-only.npz',metrics=metrics,
                by_day={'0':copy.deepcopy(metrics)},by_scene={'urban':copy.deepcopy(metrics)}))
        write(score/'metrics.json',dict(capsule_id='cap-'+pool+episode,model_seed=job['seed'],rows=rows,truth_last=True,target_feedback=False))
        write(score/'complete.json',dict(status='SCORED_COMPLETE',rows=len(rows),truth_last=True))
        write(folder/'predictions/rx/predictions.json',dict(capsule=job['capsule'],capsule_id='cap-'+pool+episode,model_seed=job['seed'],rows=[{k:v for k,v in r.items() if k not in ('metrics','by_day','by_scene')} for r in rows]))
        write(folder/'predictions/rx/query-Q1-resources.json',dict(status='COMPLETE',stage='QUERY',query_path=1,wall_seconds=2.,G_seconds=1.,F_native_feature_seconds=1.,source_encoder_seconds=None,CPU_RSS_sampled_peak=1000,CUDA_allocated_peak=2000,native_optimizer_steps=0))
        if episode.startswith('MIX'):
            common=copy.deepcopy(rows)
            for r in common:
                r.pop('by_day');r.pop('by_scene');r['metrics']['count']=(6+r['new_count'])*40
                r['metrics']['class_counts']=[40]*len(r['registered_classes'])
                for name in analysis.METRICS:
                    if r['metrics'][name] is not None:r['metrics'][name]-=.1
            write(score/'common_query_metrics.json',dict(rows=common,truth_last=True,target_feedback=False,common_query_counts={'0':240,'20':1040}))
    for seed in rt['model_seeds']:
        write(Path(rt['run_root'])/'sources'/f's{seed}'/'source_selection.json',dict(status='SOURCE_FROZEN',source_new_optimizer_updates=0,
            inherited_source_metric_training=[dict(held_RX=None,C4={'steps':600},C8={'steps':600})]))
    return rt,jobs


def test_complete_aggregation_keeps_scopes_pairs_and_seed_SD(tmp_path,monkeypatch):
    rt,jobs=fixture(tmp_path,monkeypatch);out=tmp_path/'analysis';result=analysis.analyze(rt,out)
    assert result['status']=='ANALYSIS_COMPLETE'
    summary=read(out/'summary.json');assert summary['completed_tasks']==3 and summary['MIX_units_per_KN']==2
    rows=summary['main_K5N20']['row_aggregates']
    natural=next(r for r in rows if r['dataset']=='natural' and r['arm']=='R42' and r['input_path']=='S0Q0')
    common=next(r for r in rows if r['dataset']=='common_query' and r['arm']=='R42' and r['input_path']=='S0Q0')
    h=natural['metrics']['H_macro'];assert h['unit_equal_weight']['mean']==pytest.approx(.7)
    assert h['seed_mean_statistics']['sample_SD']==pytest.approx(math.sqrt(.02))
    assert common['metrics']['H_macro']['unit_equal_weight']['mean']==pytest.approx(.6)
    pair=next(r for r in summary['main_K5N20']['paired_comparisons'] if r['dataset']=='natural' and r['comparison']=='matched_original' and r['input_path']=='S0Q0')
    assert pair['matched_input'] and pair['mean_at_least_5pp'] and pair['units']==pair['unit_wins']==pair['unit_at_least_5pp']==2
    assert pair['delta_pp']['H_macro']['unit_equal_weight']['mean']==pytest.approx(8)
    assert all(not r['matched_input'] for r in summary['paired_comparisons'] if r['comparison'].startswith('fusion_'))
    b=next(r for r in summary['row_aggregates'] if r['stage']=='B')
    assert b['metrics']['H_macro']['unit_equal_weight']==dict(n=0,mean=None,sample_SD=None)
    assert any(r['stratum']=='by_day' for r in summary['row_aggregates']) and summary['class_recall']
    assert summary['source_cost']['new_optimizer_updates']==0
    assert summary['source_cost']['per_seed'][0]['inherited_deployed_encoder_optimizer_updates']==1200
    assert summary['resources']['FLOPs'] is None and summary['resources']['shared_GPU_not_isolated']
    assert summary['resources']['groups'][0]['null_fields']==['source_encoder_seconds']
    assert {p.name for p in out.iterdir()}=={'summary.json','summary.csv','paired.csv','class_recall.csv','report.md'}
    with pytest.raises(ValueError,match='already exists'):analysis.analyze(rt,out)


def test_missing_scores_return_notcomplete_without_success_artifacts(tmp_path,monkeypatch):
    rt,jobs=fixture(tmp_path,monkeypatch)
    (Path(rt['run_root'])/'tasks/j2/scores/rx/complete.json').unlink()
    out=tmp_path/'analysis';result=analysis.analyze(rt,out)
    assert result['status']=='NOT_COMPLETE' and result['completed_tasks']==2 and not out.exists()


def test_missing_source_cost_metadata_is_unknown_not_zero(tmp_path):
    rt=dict(run_root=str(tmp_path),model_seeds=[1,2])
    write(tmp_path/'sources/s1/source_selection.json',dict(status='SOURCE_FROZEN',inherited_source_metric_training=[dict(C4={'steps':600},C8={'steps':600})]))
    result=analysis._source_cost(rt)
    assert result['new_optimizer_updates'] is None
    assert result['per_seed'][0]['inherited_source_training_optimizer_updates']==1200
    assert result['per_seed'][0]['inherited_deployed_encoder_optimizer_updates'] is None
    assert result['per_seed'][1]['source_new_optimizer_updates'] is None


@pytest.mark.parametrize('fault',['duplicate_row','duplicate_unit','missing_row','common_count','score_seed'])
def test_inconsistent_units_or_rows_are_rejected(tmp_path,monkeypatch,fault):
    rt,jobs=fixture(tmp_path,monkeypatch);score=Path(rt['run_root'])/'tasks/j1/scores/rx';path=score/'metrics.json';p=read(path)
    if fault=='duplicate_row':p['rows'].append(copy.deepcopy(p['rows'][0]))
    elif fault=='missing_row':p['rows'].pop()
    elif fault=='score_seed':p['model_seed']=9
    elif fault=='duplicate_unit':jobs.append(dict(jobs[0],key='other'))
    else:
        path=score/'common_query_metrics.json';p=read(path);p['rows'][0]['metrics']['count']-=1
    write(path,p,exclusive=False)
    with pytest.raises(ValueError):analysis.analyze(rt,tmp_path/'analysis')
    assert not (tmp_path/'analysis').exists()
