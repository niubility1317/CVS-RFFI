import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from score_d92_confirmation import score


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data),encoding='utf-8')


def fixture(tmp, sfhead=False):
    candidate_folder='sgjoint' if sfhead=='sgjoint' else ('sfhead' if sfhead else 'scv')
    candidate_method='D92-SGJoint-v1' if sfhead=='sgjoint' else ('D92-SFHead-v1' if sfhead else 'D92-SCV-v1')
    candidate_mode='d92_sgjoint_registration' if sfhead=='sgjoint' else ('d92_sfhead_registration' if sfhead else 'd92_scv_registration')
    if sfhead=='bnna':
        candidate_folder,candidate_method,candidate_mode='bnna','D92-BNNA-v1','d92_bnna_registration'
    cap=tmp/'capsule';(cap/'splits').mkdir(parents=True)
    ids=[f'physical-{i}' for i in range(16)]
    np.savez(cap/'received.npz',ids=ids)
    write(cap/'manifest.json',dict(capsule_id='synthetic',split_count=2))
    records=[]
    for n in (6,8):
        classes=[f'class-{i}' for i in range(n)]
        split=dict(split_id=f'split{n}',capsule_id='synthetic',receiver='rx',scenario='scene',k=1,support_seed=41,
            registered_classes=classes,query_indices=list(range(n)))
        write(cap/'splits'/f'split{n}.json',split)
        records.append(dict(**{k:split[k] for k in ['split_id','capsule_id','receiver','scenario','k','support_seed']},
            classes=classes,query_ids=ids[:n],predicted_indices=list(range(n)),scores=np.eye(n).tolist(),mode='d92_registration'))
    root=tmp/'run';row=root/'seed';scv=row/candidate_folder;scv.mkdir(parents=True)
    write(root/'state.json',dict(seed=dict(status='PREDICTIONS_COMPLETE')))
    baseline=records+[dict(records[0],mode='frozen_dg')]
    candidate=[dict(r,mode=candidate_mode) for r in records]
    for folder,data in [(row,baseline),(scv,candidate)]:
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data),encoding='utf-8')
        write(folder/'predictions_complete.json',dict(status='PREDICTIONS_COMPLETE',split_count=2,capsule_id='synthetic',predictions=len(data)))
    truth={ids[i]:dict(pool_role='query',receiver='rx',scene='scene',transmitter=f'class-{i}',old=i<6) for i in range(8)}
    write(tmp/'truth.json',truth)
    spec=dict(execution=dict(remote_run_root=str(root)),confirmation=dict(capsule=str(cap),truth=str(tmp/'truth.json')),
        rows=[dict(row_id='seed',output_root=str(row),seeds=dict(model=41))])
    if sfhead:
        spec['confirmation'].update(candidate_method=candidate_method, candidate_folder=candidate_folder,
            candidate_predictor='predict_d92_summary_joint.py' if sfhead=='sgjoint' else 'predict_d92_sourcefree_head.py', candidate_mode=candidate_mode,
            expected_split_count=2)
        if sfhead=='bnna':spec['confirmation']['candidate_predictor']='predict_d92_bnna.py'
    return spec,row,scv


@pytest.mark.parametrize('sfhead', [False, True, 'sgjoint', 'bnna'])
def test_joint_scoring_h_floors_f1_and_same_row_forgetting(tmp_path,sfhead):
    spec,_,_=fixture(tmp_path,sfhead)
    score(spec,tmp_path/'result.json');result=json.loads((tmp_path/'result.json').read_text())
    joint=[r for r in result['results'] if r['new_count']]
    assert len(joint)==2
    assert {r['method'] for r in joint} == {'D92', spec['confirmation'].get('candidate_method','D92-SCV-v1')}
    for r in joint:
        assert all(r[k]==1 for k in ['old_accuracy','new_accuracy','harmonic_mean','macro_f1','old_floor','new_floor'])
        assert r['forgetting']==0
    assert result['selection_feedback_forbidden']


@pytest.mark.parametrize('sfhead', [False, True, 'sgjoint', 'bnna'])
@pytest.mark.parametrize('fault',['missing','argmax','identity','duplicate','state','mode'])
def test_all_predictions_validated_before_truth_is_opened(tmp_path,fault,sfhead):
    spec,row,scv=fixture(tmp_path,sfhead);spec['confirmation']['truth']=str(tmp_path/'NEVER_OPEN_MISSING_TRUTH.json')
    path=scv/'predictions.jsonl';data=[json.loads(s) for s in path.read_text().splitlines()]
    if fault=='missing':data.pop()
    if fault=='duplicate':data.append(data[0])
    if fault=='argmax':data[0]['predicted_indices'][0]=2
    if fault=='identity':data[0]['query_ids'][0]='different'
    if fault=='mode':data[0]['mode']='d92_registration'
    if fault=='state':write(Path(spec['execution']['remote_run_root'])/'state.json',dict(seed=dict(status='RUNNING')))
    path.write_text(''.join(json.dumps(r)+'\n' for r in data))
    with pytest.raises(ValueError):score(spec,tmp_path/'out.json')
    assert not (tmp_path/'out.json').exists()


@pytest.mark.parametrize('key,value', [
    ('candidate_method', 'D92-SCV-v1'), ('candidate_folder', '../escape'),
    ('candidate_predictor', 'other.py'), ('candidate_mode', 'wrong'),
    ('expected_split_count', 300),
])
def test_scorer_rejects_invalid_alias_or_split_count_before_truth(tmp_path,key,value):
    spec,_,_=fixture(tmp_path,True)
    spec['confirmation'][key]=value
    spec['confirmation']['truth']=str(tmp_path/'NEVER_OPEN_MISSING_TRUTH.json')
    with pytest.raises(ValueError):score(spec,tmp_path/'out.json')
    assert not (tmp_path/'out.json').exists()


@pytest.mark.parametrize('sfhead', [False, True, 'sgjoint', 'bnna'])
@pytest.mark.parametrize('wrong_tie', [False, True])
def test_exact_tie_uses_only_the_preregistered_method_policy(tmp_path, sfhead, wrong_tie):
    spec, row, candidate = fixture(tmp_path, sfhead)
    # Swap two old-class physical names while retaining each stored column and score.
    # All split/record/truth registries move together; no scoring data are read by fit.
    def renamed(name):
        return {'class-0': 'class-1', 'class-1': 'class-0'}.get(name, name)
    capsule = Path(spec['confirmation']['capsule'])
    for p in (capsule / 'splits').glob('*.json'):
        v = json.loads(p.read_text())
        v['registered_classes'] = [renamed(c) for c in v['registered_classes']]
        write(p, v)
    truth_path = Path(spec['confirmation']['truth'])
    truth = json.loads(truth_path.read_text())
    for value in truth.values():
        value['transmitter'] = renamed(value['transmitter'])
    write(truth_path, truth)
    for folder in (row, candidate):
        p = folder / 'predictions.jsonl'
        values = [json.loads(line) for line in p.read_text().splitlines()]
        for value in values:
            value['classes'] = [renamed(c) for c in value['classes']]
            if folder == candidate:
                value['scores'][0] = [1.0] * len(value['classes'])
                expected = 1 if sfhead in ('sgjoint','bnna') else 0
                value['predicted_indices'][0] = (1 - expected) if wrong_tie else expected
        p.write_text(''.join(json.dumps(v) + '\n' for v in values), encoding='utf-8')
    if wrong_tie:
        spec['confirmation']['truth'] = str(tmp_path / 'NEVER_OPEN_MISSING_TRUTH.json')
        with pytest.raises(ValueError, match='argmax'):
            score(spec, tmp_path / 'out.json')
        assert not (tmp_path / 'out.json').exists()
    else:
        score(spec, tmp_path / 'out.json')
        assert json.loads((tmp_path / 'out.json').read_text())['status'] == 'SCORED'
