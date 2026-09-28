import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from score_d92_confirmation import score


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data),encoding='utf-8')


def fixture(tmp):
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
    root=tmp/'run';row=root/'seed';scv=row/'scv';scv.mkdir(parents=True)
    write(root/'state.json',dict(seed=dict(status='PREDICTIONS_COMPLETE')))
    baseline=records+[dict(records[0],mode='frozen_dg')]
    candidate=[dict(r,mode='d92_scv_registration') for r in records]
    for folder,data in [(row,baseline),(scv,candidate)]:
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data),encoding='utf-8')
        write(folder/'predictions_complete.json',dict(status='PREDICTIONS_COMPLETE',split_count=2,capsule_id='synthetic',predictions=len(data)))
    truth={ids[i]:dict(pool_role='query',receiver='rx',scene='scene',transmitter=f'class-{i}',old=i<6) for i in range(8)}
    write(tmp/'truth.json',truth)
    spec=dict(execution=dict(remote_run_root=str(root)),confirmation=dict(capsule=str(cap),truth=str(tmp/'truth.json')),
        rows=[dict(row_id='seed',output_root=str(row),seeds=dict(model=41))])
    return spec,row,scv


def test_joint_scoring_h_floors_f1_and_same_row_forgetting(tmp_path):
    spec,_,_=fixture(tmp_path)
    score(spec,tmp_path/'result.json');result=json.loads((tmp_path/'result.json').read_text())
    joint=[r for r in result['results'] if r['new_count']]
    assert len(joint)==2
    for r in joint:
        assert all(r[k]==1 for k in ['old_accuracy','new_accuracy','harmonic_mean','macro_f1','old_floor','new_floor'])
        assert r['forgetting']==0
    assert result['selection_feedback_forbidden']


@pytest.mark.parametrize('fault',['missing','argmax','identity','duplicate','state'])
def test_all_predictions_validated_before_truth_is_opened(tmp_path,fault):
    spec,row,scv=fixture(tmp_path);spec['confirmation']['truth']=str(tmp_path/'NEVER_OPEN_MISSING_TRUTH.json')
    path=scv/'predictions.jsonl';data=[json.loads(s) for s in path.read_text().splitlines()]
    if fault=='missing':data.pop()
    if fault=='duplicate':data.append(data[0])
    if fault=='argmax':data[0]['predicted_indices'][0]=2
    if fault=='identity':data[0]['query_ids'][0]='different'
    if fault=='state':write(Path(spec['execution']['remote_run_root'])/'state.json',dict(seed=dict(status='RUNNING')))
    path.write_text(''.join(json.dumps(r)+'\n' for r in data))
    with pytest.raises(ValueError):score(spec,tmp_path/'out.json')
    assert not (tmp_path/'out.json').exists()
