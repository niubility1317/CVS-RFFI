"""Explicit synthetic owner metadata, never copied from a historical spec."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import prepare_d92_conditional_joint_probe as prepare
import run_d92_conditional_joint_probe as run


def request(nrows=1,pairs=1):
    old=['old-'+str(i) for i in range(6)];selected_pairs=[['synthetic-rx'+str(i),'practical_high'] for i in range(pairs)]
    splits=[]
    for index,(rx,scene,k,new) in enumerate(itertools.product([p[0] for p in selected_pairs],['practical_high'],run.KS,run.NEW_COUNTS)):
        splits.append(dict(split_id='synthetic-'+str(index),receiver=rx,scenario=scene,k=k,new_count=new,
            support_seed=7,registered_classes=old+['new-'+str(i) for i in range(new)]))
    selection=dict(receiver_scenes=selected_pairs,support_seed=7,ks=run.KS,new_counts=run.NEW_COUNTS,splits=splits)
    matrix=dict(receivers=[p[0] for p in selected_pairs],scenarios=run.SCENARIOS,ks=run.KS,new_counts=run.NEW_COUNTS,support_seeds=[7])
    return dict(run_id='synthetic-conditional-run',group_id='synthetic',spec_path='configs/synthetic-conditional.json',
        code=dict(cwd='/synthetic/release',environment='/synthetic/python'),
        execution=dict(remote_run_root='/synthetic/run',launch_owner='root',cpu_lanes=2,blas_threads_per_lane=2),
        permissions=dict(query_use='none; query IQ/labels/truth/scores never read',source_samples=False,source_per_record_features=False,
            summary_inputs=False,old_target_scores_for_adaptation=False,cross_run_result_tuning=False,
            adapted_state_reuse='within_path_B_to_C_only; no_cross_parent_fold_or_model_reuse'),
        probe=dict(cohorts=dict(synthetic=dict(capsule='/synthetic/capsule',capsule_id='residual-noeq-synthetic',
            config_path='configs/synthetic-evaluator.json',matrix=matrix,selection=selection,expected_split_count=60*pairs))),
        rows=[dict(row_id='row-'+str(i),cohort='synthetic',support_features='/synthetic/cache-'+str(i),expected_checkpoint_sha256=format(i+1,'064x'),
            seeds=dict(model=i,split=None,data=None,augmentation=None,support=7,evaluation=None)) for i in range(nrows)])


def spec(nrows=1,pairs=1):
    req=request(nrows,pairs)
    return prepare.documents(req,commit='a'*40)[req['spec_path']]


def test_explicit_request_preserved_and_derived_config_exclusive(tmp_path):
    original=request(3);before=deepcopy(original);docs=prepare.documents(original,commit='a'*40)
    assert original==before and len(docs)==2
    value=docs[original['spec_path']];run.validate_spec(value)
    assert value['actual_A'] is value['adaptation_gain_B_minus_A'] is value['performance_gate'] is None
    assert len(value['rows'])==3 and value['probe']['budget']['total']['exact']['episodes']==60
    for actual,expected in zip(value['rows'],original['rows']):
        assert actual['seeds']==expected['seeds'] and actual['expected_checkpoint_sha256']==expected['expected_checkpoint_sha256']
    prepare.write_documents(tmp_path,docs)
    assert json.loads((tmp_path/original['spec_path']).read_text(encoding='utf-8'))==value
    with pytest.raises(FileExistsError):prepare.write_documents(tmp_path,docs)


def test_no_hidden_old_spec_defaults_and_no_partial_overwrite(tmp_path):
    value=request();value['rows'][0]['seeds'].pop('augmentation')
    with pytest.raises(ValueError):prepare.documents(value,commit='a'*40)
    with pytest.raises(ValueError):prepare.documents(request(),commit='unknown')
    with pytest.raises(ValueError):prepare.write_documents(tmp_path,{'../escaped.json':{}})
    (tmp_path/'present.json').write_text('preserve',encoding='utf-8')
    with pytest.raises(FileExistsError):prepare.write_documents(tmp_path,{'new.json':{},'present.json':{}})
    assert not (tmp_path/'new.json').exists() and (tmp_path/'present.json').read_text()=='preserve'
