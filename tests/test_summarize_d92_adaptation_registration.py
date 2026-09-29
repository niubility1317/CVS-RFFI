"""Full synthetic identity/score matrices; never historical target artifacts."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import summarize_d92_adaptation_registration as tool


def inputs(method='D92-BranchLocalRidge-v1'):
    specs=[];scored=[];identities=[];old=[f'class-{i}' for i in range(6)]
    for cohort,receivers in enumerate((['r1','r2','r3'],['r4'])):
        cid=f'capsule-{cohort}';run=f'synthetic-{cohort}'
        data=dict(target_receivers=receivers,scenarios=['s1','s2','s3'],k=[1,5,10,20],
            new_class_counts=[0,2,5,10,20],support_seeds=[11,12,13,14,15],capsule_id=cid)
        rows=[dict(row_id=f'model-{seed}',seeds=dict(model=seed),source_root=f'/source/{seed}',
            reuse_row_root=f'/baseline/{cohort}/{seed}',expected_checkpoint_sha256=f'{seed:064x}')
            for seed in range(2026092701,2026092705)]
        folder=tool.METHODS[method]
        spec=dict(run_id=run,execution=dict(remote_run_root='/runs/'+run),rows=rows,data=data,
            confirmation=dict(candidate_method=method,candidate_folder=folder,
                candidate_predictor='evaluate_d92_'+folder+'.py',candidate_mode='d92_'+folder+'_registration',
                old_classes=old,reuse_validated_capsule_id=cid,expected_split_count=300*len(receivers),
                splits_per_model=300*len(receivers),predictions_total=2412*len(receivers)))
        splits=[];dg=[]
        for rx,scene,k,new,support in product(receivers,data['scenarios'],data['k'],data['new_class_counts'],data['support_seeds']):
            classes=[f'class-{i}' for i in range(6+new)]
            split=dict(split_id=f'{rx}-{scene}-{k}-{new}-{support}',capsule_id=cid,receiver=rx,scenario=scene,
                k=k,support_seed=support,registered_classes=classes,
                support_ids=[f'{rx}-{scene}-support-{support}-{cls}-{j}' for cls in classes for j in range(k)],
                support_labels=[i for i in range(len(classes)) for _ in range(k)],
                query_ids=[f'{rx}-{scene}-query-{cls}' for cls in classes])
            splits.append(split)
            if k==1 and new==0 and support==11:
                dg.append(dict(split_id=split['split_id'],capsule_id=cid,receiver=rx,scenario=scene,
                    classes=classes,query_ids=split['query_ids'],k=999,support_seed=-999))
        models=[];records=[]
        for source in rows:
            seed=source['seeds']['model']
            models.append(dict(model_seed=seed,row_id=source['row_id'],source_root=source['source_root'],
                reuse_row_root=source['reuse_row_root'],checkpoint_sha256=source['expected_checkpoint_sha256'],
                splits=splits,frozen_dg=dg))
            for split in splits:
                new=len(split['registered_classes'])-6
                common=dict(row_id=source['row_id'],model_seed=seed,split_id=split['split_id'],
                    receiver=split['receiver'],scenario=split['scenario'],k=split['k'],new_count=new,
                    support_seed=split['support_seed'],classes=split['registered_classes'],
                    class_count=len(split['registered_classes']),query_count=len(split['query_ids']))
                bv=.55 if cohort==0 else .75;cv=bv if not new else (.54 if cohort==0 else .72)
                nv=None if not new else (.52 if cohort==0 else .76)
                h=None if not new else 2*cv*nv/(cv+nv)
                records.append(dict(common,method=method,old_accuracy=cv,new_accuracy=nv,harmonic_mean=h))
                records.append(dict(common,method='D92',old_accuracy=.4,new_accuracy=None if not new else .4,
                                    harmonic_mean=None if not new else .4))
            for original in dg:
                records.append(dict(row_id=source['row_id'],model_seed=seed,method='frozen_dg',
                    **{key:original[key] for key in ('split_id','receiver','scenario','k','support_seed','classes')},
                    new_count=0,class_count=6,query_count=6,old_accuracy=.4,new_accuracy=None,harmonic_mean=None))
        specs.append(spec);scored.append(dict(status='SCORED',selection_feedback_forbidden=True,results=records))
        identities.append(dict(schema=tool.IDENTITY_SCHEMA,run_id=run,capsule_id=cid,candidate_method=method,models=models))
    return specs,scored,identities


@pytest.fixture(scope='module')
def frozen_inputs():
    return inputs()


@pytest.fixture
def ready(frozen_inputs):
    return deepcopy(frozen_inputs)


@pytest.mark.parametrize('method',list(tool.METHODS))
def test_full_abc_matrix_equal_cells_and_no_decision(method):
    args=inputs(method);before=deepcopy(args);result=tool.analyze(*args)
    assert args==before
    assert result['analysis_only'] is True and result['automatic_promotion'] is False
    assert result['selected_candidate'] is None and result['selection_feedback_forbidden'] is True
    assert len(result['cells'])==result['combined']['cells']==4800
    assert 'preregistered_guard_pass' not in result and 'advancement_gate' not in result
    per_k=result['combined']['tables']['per_k']
    for row in per_k:
        assert row['cells']==960
        assert row['A_old_accuracy']==pytest.approx(.4)
        assert row['B_old_accuracy']==pytest.approx(.6)
        assert row['C_old_accuracy']==pytest.approx(.585)
        assert row['C_new_accuracy']==pytest.approx(.58)
        assert row['adaptation_gain_pp']==pytest.approx(20.)
        assert row['registration_old_drop_pp']==pytest.approx(1.5)
        assert row['old_new_gap_pp']==pytest.approx(2.5)  # Mean abs; not abs of means (.5).
        expected_h=.75*(2*.54*.52/(.54+.52))+.25*(2*.72*.76/(.72+.76))
        assert row['C_harmonic_mean']==pytest.approx(expected_h)
        assert row['descriptive_ideal_status']==dict(adaptation_at_least_10pp=True,
            registration_drop_at_most_1pp=False,old_new_gap_at_most_3pp=True,all_three_met=False)
        assert row['descriptive_ideal_distance_pp']['registration_drop_excess_pp']==pytest.approx(.5)
    assert all(r['cells']==1200 for r in result['combined']['tables']['per_k_all'])
    assert all(r['cells']==240 for r in result['combined']['tables']['per_k_new'])
    assert len(result['combined']['tables']['per_model_k'])==16
    assert len(result['combined']['tables']['per_receiver_scene_k'])==48
    assert 'replacing' in result['claim_scope'] and 'neither assumed nor required' in result['claim_scope']


def test_new_zero_metrics_and_descriptive_distance(ready):
    result=tool.analyze(*ready)
    for row in result['combined']['tables']['per_k_new']:
        if row['new_count']==0:
            assert row['B_old_accuracy']==row['C_old_accuracy']
            assert row['registration_old_drop_pp']==0
            assert row['C_new_accuracy'] is row['old_new_gap_pp'] is row['C_harmonic_mean'] is None
            assert row['descriptive_ideal_status']['all_three_met'] is None
            assert row['descriptive_ideal_distance_pp']['old_new_gap_excess_pp'] is None
    row=result['combined']['tables']['overall_all'][0]
    assert row['defined_cells']['A_old_accuracy']==4800 and row['defined_cells']['C_new_accuracy']==3840


def test_ideal_thresholds_are_only_descriptive_and_negative_drop_is_valid():
    c=tool.cell_measures(dict(old_accuracy=.5,new_accuracy=None,harmonic_mean=None),
        dict(old_accuracy=.6,new_accuracy=None,harmonic_mean=None),
        dict(old_accuracy=.59,new_accuracy=.62,harmonic_mean=2*.59*.62/1.21),2)
    assert c['descriptive_ideal_status']['all_three_met'] is True
    bad=tool.cell_measures(dict(old_accuracy=.9,new_accuracy=None,harmonic_mean=None),
        dict(old_accuracy=.1,new_accuracy=None,harmonic_mean=None),
        dict(old_accuracy=.2,new_accuracy=.9,harmonic_mean=2*.2*.9/1.1),2)
    assert bad['registration_old_drop_pp']==pytest.approx(-10)
    assert bad['descriptive_ideal_status']['all_three_met'] is False  # Still produces a report.


@pytest.mark.parametrize('fault',['missing_cohort','source','sha','method','missing_matrix','duplicate_run',
    'overlap_rx','identity_source','identity_capsule','missing_model','old_support','old_query','extra_payload'])
def test_binding_and_physical_pairing_fail_with_no_metric_aggregation(ready,monkeypatch,fault):
    specs,scored,identities=ready
    if fault=='missing_cohort': specs.pop()
    elif fault=='source': specs[1]['rows'][0]['source_root']='/different'
    elif fault=='sha': specs[1]['rows'][0]['expected_checkpoint_sha256']='b'*64
    elif fault=='method': specs[1]['confirmation']['candidate_method']='D92'
    elif fault=='missing_matrix': specs[1]['data']['k']=[1,5,10]
    elif fault=='duplicate_run': specs[1]['run_id']=specs[0]['run_id']
    elif fault=='overlap_rx': specs[1]['data']['target_receivers']=['r1']
    elif fault=='identity_source': identities[1]['models'][0]['source_root']='/wrong'
    elif fault=='identity_capsule': identities[1]['capsule_id']='wrong'
    elif fault=='missing_model': identities[1]['models'].pop()
    else:
        split=next(s for s in identities[1]['models'][0]['splits'] if len(s['registered_classes'])>6)
        if fault=='old_support': split['support_ids'][0]='different-physical-support'
        elif fault=='old_query': split['query_ids'][0]='different-old-query'
        else: split['scores']=[.5]
    monkeypatch.setattr(tool,'cell_measures',lambda *a,**k:pytest.fail('Metric aggregation before full identity barrier'))
    with pytest.raises(ValueError):tool.analyze(specs,scored,identities)


@pytest.mark.parametrize('fault',['missing_score','duplicate_score','row_id','split_id','query_count','class_order',
    'extra_dg','checkpoint','nonfinite','out_of_range','old_only_h','wrong_h'])
def test_score_coverage_identity_and_metric_validation(ready,fault):
    specs,scored,identities=ready;records=scored[-1]['results'];row=next(r for r in records if r['method']==specs[-1]['confirmation']['candidate_method'] and r['new_count']>0)
    if fault=='missing_score':records.pop()
    elif fault=='duplicate_score':records.append(deepcopy(records[0]))
    elif fault=='row_id':row['row_id']='wrong'
    elif fault=='split_id':row['split_id']='wrong'
    elif fault=='query_count':row['query_count']+=1
    elif fault=='class_order':row['classes']=row['classes'][::-1]
    elif fault=='extra_dg':records.append(deepcopy(next(r for r in records if r['method']=='frozen_dg')))
    elif fault=='checkpoint':row['checkpoint_sha256']='b'*64
    elif fault=='nonfinite':row['old_accuracy']=float('nan')
    elif fault=='out_of_range':row['new_accuracy']=1.1
    elif fault=='old_only_h':next(r for r in records if r['method']=='frozen_dg')['harmonic_mean']=.3
    else:row['harmonic_mean']=.99
    with pytest.raises(ValueError):tool.analyze(specs,scored,identities)


def write_inputs(tmp_path,args):
    folders=[];paths=[]
    for i,(spec,data,identity) in enumerate(zip(*args)):
        folder=tmp_path/f'cohort-{i}';folder.mkdir();folders.append(folder)
        for name,value in [('startup',dict(spec=spec,commit='runtime')),('complete',dict(status='SCORED',commit='runtime',records=len(data['results']))),('scores',data)]:
            (folder/(name+'.json')).write_text(json.dumps(value),encoding='utf-8')
        path=tmp_path/f'identity-{i}.json';path.write_text(json.dumps(identity),encoding='utf-8');paths.append(path)
    return folders,paths


@pytest.mark.parametrize('fault',['running','commit','count','source','identity_source','physical'])
def test_all_metadata_barriers_precede_any_score_file_access(tmp_path,ready,monkeypatch,fault):
    folders,paths=write_inputs(tmp_path,ready)
    if fault in ('running','commit','count'):
        path=folders[-1]/'complete.json';value=tool.read(path)
        if fault=='running':value['status']='RUNNING'
        elif fault=='commit':value['commit']='wrong'
        else:value['records']-=1
    elif fault=='source':
        path=folders[-1]/'startup.json';value=tool.read(path);value['spec']['rows'][0]['source_root']='/wrong'
    else:
        path=paths[-1];value=tool.read(path)
        if fault=='identity_source':value['models'][0]['source_root']='/wrong'
        else:next(s for s in value['models'][0]['splits'] if len(s['registered_classes'])>6)['query_ids'][0]='wrong-old-query'
    path.write_text(json.dumps(value),encoding='utf-8')
    original=Path.open
    def guard(path,*a,**kw):
        assert path.name!='scores.json','Scores opened before both completion and identity barriers'
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guard)
    with pytest.raises(ValueError):tool.prepare(folders,paths)


def test_prepare_write_preserves_inputs_and_refuses_output_overwrite(tmp_path,ready):
    folders,paths=write_inputs(tmp_path,ready)
    files=[p for f in folders for p in f.iterdir()]+paths;before={str(p):p.read_bytes() for p in files}
    result=tool.prepare(folders,paths);output=tmp_path/'report';tool.write(result,output)
    assert before=={str(p):p.read_bytes() for p in files}
    report=(output/'report.md').read_text(encoding='utf-8')
    assert '分类头替换' in report and 'descriptive_ideal_status' in report and 'N/A' in report
    assert (output/'combined_rx4/per_k_new.csv').exists() and (output/'rx3/per_k.csv').exists() and (output/'rx1/per_k.csv').exists()
    with pytest.raises(FileExistsError):tool.write(result,output)
