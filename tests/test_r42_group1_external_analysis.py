import copy
from pathlib import Path
import pytest
from experiments.pamf_phase2_practical import r42_group1_external_analysis as analysis
from experiments.pamf_phase2_practical.io import read,write
from experiments.pamf_phase2_practical.r42_suite_worker import desired_rows


def metric(stage,count,value):
    return dict(count=count*10,old_macro_accuracy=value,new_macro_accuracy=value if stage=='C' else None,
        H_macro=value if stage=='C' else None,macro_f1=value,class_floor=value-.1,accuracy=value,macro_accuracy=value)


def fixture(tmp_path,monkeypatch):
    rt=dict(run_root=str(tmp_path/'suite'),run_id='suite',model_seeds=[1,2]);jobs=[];matrix=[]
    for seed in (1,2):
        for episode in ('MIX02','LS01'):
            task=dict(key=f'{seed}-{episode}',seed=seed,capsule=f'capsules/P01/{episode}/rx',points=[[5,20]]);matrix.append(task)
            rows=[];value=.7 if seed==1 else .8
            for r in desired_rows(task['points']):
                classes=[f'c{i}' for i in range(6+r['new_count'])]
                rows.append(dict(r,split_id=f"split-{r['new_count']}",registered_classes=classes,prediction='metadata-only.npz',
                    physical_pool='P01',episode=episode,receiver='rx',model_seed=seed,metrics=metric(r['stage'],len(classes),value if r['arm']=='R42' else .6)))
            folder=Path(rt['run_root'])/'tasks'/task['key']
            write(folder/'scores/rx/metrics.json',dict(capsule_id=episode,model_seed=seed,truth_last=True,target_feedback=False,rows=rows))
            write(folder/'scores/rx/complete.json',dict(status='SCORED_COMPLETE',rows=len(rows),truth_last=True))
            write(folder/'predictions/rx/predictions.json',dict(capsule=task['capsule'],capsule_id=episode,model_seed=seed,rows=[{k:v for k,v in r.items() if k!='metrics'} for r in rows]))
    monkeypatch.setattr(analysis,'tasks',lambda rt:matrix)
    for profile in analysis.PROFILES:
        for seed in (1,2):
            source=tmp_path/profile/f'source{seed}';complete=source/'group1_complete.json'
            write(complete,dict(status='GROUP1_SOURCE_READY',profile=profile,seed=seed,resources='A_resources.json'))
            write(source/'A_resources.json',dict(status='COMPLETE',optimizer_steps=10))
            jobs.append(dict(id=f'{profile}-source{seed}',kind='source',profile=profile,seed=seed,output=str(source),completion=str(complete)))
        for task in matrix:
            episode=task['capsule'].split('/')[2];seed=task['seed'];pred=tmp_path/profile/task['key']/'pred';score=pred.parent/'score';rows=[]
            for stage,k,n in sorted(analysis.native_rows(profile,episode)):
                count=6+n;classes=[f'c{i}' for i in range(count)];old=6*k if stage=='B' else 0;new=n*k if stage=='C' else 0
                filename=f'{stage}-K{k}-N{n}.npz';resource=dict(status='COMPLETE',optimizer_steps=2 if stage=='B' else 4 if stage=='C' else 0,
                    actual_target_old_labels=old,actual_target_new_labels=new)
                resource_name=f'B-K{k}-fit.json' if stage=='B' else f'C-K{k}-N{n}-fit.json' if stage=='C' else f'A0-K{k}.json'
                write(pred/resource_name,resource)
                rows.append(dict(stage=stage,k=k,new_count=n,arm=profile,input_path='P0',split_id=f'split-{n}',registered_classes=classes,prediction=filename,
                    physical_pool='P01',episode=episode,receiver='rx',model_seed=seed,resources=resource,metrics=metric(stage,count,.5)))
            write(pred/'predictions.json',dict(capsule=task['capsule'],capsule_id=episode,model_seed=seed,profile=profile,rows=[{k:v for k,v in r.items() if k!='metrics'} for r in rows]))
            write(pred/'complete.json',dict(status='PREDICTIONS_COMPLETE',rows=len(rows),truth_read=False,query_fit=False))
            write(score/'metrics.json',dict(capsule_id=episode,model_seed=seed,truth_last=True,target_feedback=False,rows=rows))
            write(score/'complete.json',dict(status='SCORED_COMPLETE',rows=len(rows),truth_last=True))
            jobs.append(dict(id=profile+'-predict-'+task['key'],kind='predict',profile=profile,seed=seed,capsule=task['capsule'],output=str(pred),completion=str(pred/'complete.json'),scope='full'))
            jobs.append(dict(id=profile+'-score-'+task['key'],kind='score',profile=profile,seed=seed,prediction=str(pred),output=str(score),completion=str(score/'complete.json')))
    # The stopped/unbound methods and global pipeline status never block these profiles.
    jobs.append(dict(profile='issl',kind='predict',output='missing',completion='missing'))
    return rt,matrix,dict(jobs=jobs,pipeline_state='FAILED/BLOCKED')


def test_native_stage_coverage_and_curve_work_points():
    assert len(analysis.native_rows('protonet_cda','MIX01'))==8
    assert len(analysis.native_rows('mopc_hr','MIX01'))==20
    assert {s for s,_,_ in analysis.native_rows('mopc_hr','LS01')}=={'A0','C'}
    assert {s for s,_,_ in analysis.native_rows('mrior_sda','MIX02')}=={'A0','B'}


def test_pilot_main_and_curve_extra_merge_without_duplicate_unit(tmp_path):
    task=dict(seed=1,capsule='capsules/P01/MIX01/1-1');profile='mopc_hr';parts=[];scores={}
    for scope in ('pilot_main','curve_extra'):
        pred=tmp_path/scope/'pred';score=tmp_path/scope/'score';rows=[]
        for stage,k,n in sorted(analysis.native_rows(profile,'MIX01')):
            main=(k,n) in ((5,0),(5,20))
            if main!=(scope=='pilot_main'):continue
            rows.append(dict(stage=stage,k=k,new_count=n,arm=profile,input_path='P0',split_id=f'{k}-{n}',prediction=f'{stage}-{k}-{n}.npz',
                registered_classes=[f'c{i}' for i in range(6+n)],physical_pool='P01',episode='MIX01',receiver='1-1',model_seed=1,metrics=metric(stage,6+n,.5)))
        write(pred/'predictions.json',dict(capsule=task['capsule'],capsule_id='cap',model_seed=1,profile=profile,rows=rows))
        write(score/'metrics.json',dict(capsule_id='cap',model_seed=1,truth_last=True,rows=rows))
        write(score/'complete.json',dict(status='SCORED_COMPLETE',rows=len(rows)))
        part=dict(seed=1,output=str(pred));parts.append(part);scores[1,str(pred)]=dict(output=str(score),completion=str(score/'complete.json'))
    plan=dict(predict={(1,task['capsule']):parts},score=scores)
    merged=analysis._external_rows(plan,profile,task,'cap')
    assert set(merged)==analysis.native_rows(profile,'MIX01') and len(merged)==20
    assert len([r for r in merged if r[1:]==(5,20)])==1


def test_complete_native_raw_comparison_and_label_budget_difference(tmp_path,monkeypatch):
    rt,matrix,jobs=fixture(tmp_path,monkeypatch);out=tmp_path/'analysis'
    result=analysis.analyze(rt,jobs,out);assert result['status']=='ANALYSIS_COMPLETE'
    summary=read(out/'summary.json');assert len(summary['profiles'])==10 and set(summary['profile_units'].values())=={4}
    pair=next(r for r in summary['paired_summary'] if r['profile']=='mopc_hr' and r['scope']=='main_MIX8_K5N20' and r['stratum']=='all' and r['comparator']=='R42')
    assert pair['delta_pp']['H_macro']['unit_equal_weight']['mean']==pytest.approx(25)
    assert not pair['label_equivalent'] and pair['matched_raw_input'] and pair['units']==pair['unit_wins']==2
    assert pair['delta_pp']['H_macro']['seed_mean_statistics']['sample_SD']==pytest.approx(50**.5)
    native=next(r for r in summary['native_summary'] if r['profile']=='mopc_hr' and r['stage']=='C')
    assert native['actual_old_labels']==0 and native['actual_new_labels']==100
    assert native['actual_optimizer_steps']['unit_equal_weight']['mean']==4
    assert summary['unavailable']['issl'].startswith('USER_STOPPED')
    assert all(r['input_path']=='P0' for r in summary['native_summary'])
    assert all(r['stage']!='C' for r in summary['native_summary'] if r['profile'] in analysis.DA)
    assert all(r['stage']!='B' for r in summary['native_summary'] if r['profile'] in analysis.CIL)
    assert summary['conventions']['FLOPs'] is None
    assert {p.name for p in out.iterdir()}=={'summary.json','native_summary.csv','paired_summary.csv','paired_units.csv','report.md'}


def test_missing_required_score_is_notcomplete_no_report(tmp_path,monkeypatch):
    rt,matrix,jobs=fixture(tmp_path,monkeypatch);score=next(j for j in jobs['jobs'] if j.get('profile')=='mopc_hr' and j.get('kind')=='score')
    Path(score['completion']).unlink();out=tmp_path/'analysis';result=analysis.analyze(rt,jobs,out)
    assert result['status']=='NOT_COMPLETE' and not out.exists()


@pytest.mark.parametrize('fault',['split','ambiguous_score','bad_stage','wrong_seed'])
def test_matching_and_native_role_boundaries_reject_bad_inputs(tmp_path,monkeypatch,fault):
    rt,matrix,jobs=fixture(tmp_path,monkeypatch)
    score=next(j for j in jobs['jobs'] if j.get('profile')=='mopc_hr' and j.get('kind')=='score')
    pred=Path(score['prediction']);path=Path(score['output'])/'metrics.json';payload=read(path)
    if fault=='ambiguous_score':jobs['jobs'].append(dict(score,id='other',output=str(tmp_path/'other-score')))
    elif fault=='wrong_seed':payload['model_seed']=99;write(path,payload,exclusive=False)
    else:
        index=read(pred/'predictions.json')
        if fault=='split':
            payload['rows'][1]['split_id']='other';index['rows'][1]['split_id']='other'
        else:
            row=next(r for r in payload['rows'] if r['stage']=='C');row['stage']='B'
            row=next(r for r in index['rows'] if r['stage']=='C');row['stage']='B'
        write(path,payload,exclusive=False);write(pred/'predictions.json',index,exclusive=False)
    with pytest.raises(ValueError):analysis.analyze(rt,jobs,tmp_path/'analysis')
    assert not (tmp_path/'analysis').exists()
