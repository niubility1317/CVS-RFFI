"""Synthetic trial evidence for the r02 verifier-only scalar boundary.

The numerical tests use the real support producer, not a replayed fit or an
invented trial. Root runs them in the project environment. --static-only reads
source and checks UTF-8/AST plus the single permitted clone substitution.
"""
import ast
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'tools/analyze_d92_support_metric_joint_probe.py'
REVISION=ROOT/'tools/analyze_d92_support_metric_joint_probe_scalar_r02.py'
BEFORE=b"_equal(trial['comparison_tolerance']/tolerance,1.,'Trial arithmetic tolerance changed')"
AFTER=b"_equal(float(trial['comparison_tolerance']/tolerance),1.,'Trial arithmetic tolerance changed')"


def assert_only_scalar_boundary_changed():
    before=ORIGINAL.read_bytes();after=REVISION.read_bytes()
    assert before.count(BEFORE)==1 and before.count(AFTER)==0
    assert after==before.replace(BEFORE,AFTER)


if __name__=='__main__' and sys.argv[1:]==['--static-only']:
    for file in (REVISION,Path(__file__)):
        raw=file.read_bytes();source=raw.decode('utf-8',errors='strict')
        assert not raw.startswith(b'\xef\xbb\xbf')
        assert not any(ord(c)<32 and c not in '\n\r\t' for c in source)
        ast.parse(source,filename=str(file))
    assert_only_scalar_boundary_changed()
    print('UTF8/AST PASS: r02 analyzer/test; byte-exact single float boundary change; no numerical imports')
    raise SystemExit(0)

from copy import deepcopy
import json

sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code'),str(ROOT/'tests')]
import numpy as np
import pytest
import analyze_d92_support_metric_joint_probe as original
import analyze_d92_support_metric_joint_probe_scalar_r02 as revision
import test_d92_support_metric_joint_analysis as prior_fixture


@pytest.fixture(scope='module')
def nonzero_trial_parent(tmp_path_factory):
    import evaluate_d92_support_metric_joint_probe as producer
    import test_d92_proto_frame_joint_probe as fixture
    root=tmp_path_factory.mktemp('scalar-boundary-nonzero-production')
    archive=producer.StateArchive(root);stages=[];fixed=[]
    # new0 isolates the real B trial verifier without introducing a C gate or
    # changing the old support fixture's frozen geometry/resources.
    data=fixture.synthetic(k=3,new=0,zero=False)
    assert any(np.count_nonzero(data[name]) for name in producer.BRANCHES)
    def log(value):
        stages.append(producer.json_native(dict(producer.compact_event(value),
            schema=producer.SCHEMA,method=producer.METHOD,split_id='literal')))
    result=producer.probe_support_metric_joint(**data,**prior_fixture.RESOURCES,
        prototype_frame=fixture.frame(),ground_head=fixture.real_ground(),
        context=dict(run_id='literal-run',row_id='literal-row',split_id='literal'),
        state_callback=archive,log_callback=log,prediction_callback=fixed.append)
    archive.finalize('COMPLETE')
    identity=dict(split_id='literal',receiver='receiver',scenario='scene',k=3,
        support_seed=7,new_count=0,registered_classes=data['classes'])
    parent=producer.json_native(dict(result,**{key:identity[key] for key in revision.IDENTITY},
        scope=revision.SCOPE,query_rows_used=0,source_rows_used=0))
    paths=parent['folds']+parent['oneshot_proxy']['trials']
    nonzero=[path for path in paths if path['candidate_stages'][0]['trial_count']>0]
    assert nonzero, 'The production fixture must execute a real nonzero trial'
    assert parent['trial_count']>0
    assert all(path['candidate_stages'][0]['gradient_norm']>0 for path in nonzero)
    assert all(path['candidate_stages'][0]['direction_norm']>0 for path in nonzero)
    assert all(path['c_reuses_b_candidates'] for path in paths)
    return dict(root=root,parent=parent,identity=identity,physical=data['support_ids'],
        stages=stages,fixed=fixed,trial_path=nonzero[0],ground=dict(
            status='MATCHED_SOURCE_ONLY_PACKET',ordered_classes=list(reversed(prior_fixture.OLD))))


def verify_parent(module,evidence):
    archives=module.Archives(evidence['root'],'literal-run','literal-row')
    stages=iter(evidence['stages']);fixed=iter(evidence['fixed'])
    value=module.verify_parent(evidence['parent'],evidence['identity'],evidence['physical'],
        row_id='literal-row',run_id='literal-run',archives=archives,stage_stream=stages,
        prediction_stream=fixed,resources=prior_fixture.RESOURCES,ground_binding=evidence['ground'])
    archives.close()
    assert next(stages,None) is None and next(fixed,None) is None
    return value


def verify_trial_stage(module,evidence,stage):
    """Use the actual physical archive and independent complete stage checks.

    Negative tests change only a recorded scalar in a copy of the stage. No
    training function or archived numeric state is changed or rerun here.
    """
    archives=module.Archives(evidence['root'],'literal-run','literal-row')
    module.verify_row_basis(evidence['parent']['row_basis_audit'],archives,prior_fixture.RESOURCES)
    path=evidence['trial_path']
    coords=module._coordinates('literal-row','literal-run',evidence['parent'],path)
    labels={item['physical_id']:item['class_id'] for item in evidence['parent']['physical_fold_assignment']}
    return module._verify_stage(stage,path['preparations'][0],path,coords,archives,
        prior_fixture.RESOURCES,label_mapping=labels)


def test_byte_exact_clone_has_one_scalar_boundary_change():
    assert_only_scalar_boundary_changed()
    assert original.EPS.tobytes()==revision.EPS.tobytes()==np.finfo(np.float64).eps.tobytes()
    assert original.ALGORITHM==revision.ALGORITHM


def test_real_nonzero_trial_old_type_failure_new_full_verification(nonzero_trial_parent):
    stage=nonzero_trial_parent['trial_path']['candidate_stages'][0]
    assert stage['trial_count']>0 and stage['direction_norm']>0
    with pytest.raises(ValueError,match='Trial arithmetic tolerance changed'):
        verify_parent(original,nonzero_trial_parent)
    verified=verify_parent(revision,nonzero_trial_parent)
    assert verified['counters']['trial_count']==nonzero_trial_parent['parent']['trial_count']>0
    assert verified['counters']['candidate_stage_count']==6


def test_recorded_tolerance_keeps_exact_binary64_formula(nonzero_trial_parent):
    # Rebuild the core's scalar formula from its persisted operands. This uses
    # actual trial records; the clone byte check separately locks all constants.
    stage=nonzero_trial_parent['trial_path']['candidate_stages'][0]
    assert stage['trial_count']>0
    for trial in stage['trials']:
        expected=128*np.finfo(np.float64).eps*max(1.,abs(trial['loss_before']),
            abs(trial['loss_after']),abs(trial['armijo_rhs']))
        recorded=np.float64(trial['comparison_tolerance'])
        assert recorded.tobytes()==np.float64(expected).tobytes()
        ratio=trial['comparison_tolerance']/expected
        assert type(ratio) is np.float64
        assert ratio==1.
        with pytest.raises(ValueError,match='Scalar type boundary'):
            original._equal(ratio,1.,'Scalar type boundary')
        revision._equal(float(ratio),1.,'Scalar type boundary')


@pytest.mark.parametrize('invalid',['zero','negative','double','bool','string'])
def test_invalid_trial_tolerance_remains_rejected(nonzero_trial_parent,invalid):
    stage=deepcopy(nonzero_trial_parent['trial_path']['candidate_stages'][0])
    trial=stage['trials'][0];tol=trial['comparison_tolerance']
    trial['comparison_tolerance']={'zero':0.,'negative':-tol,'double':2*tol,
        'bool':True,'string':str(tol)}[invalid]
    with pytest.raises((ValueError,TypeError)):
        verify_trial_stage(revision,nonzero_trial_parent,stage)


@pytest.mark.parametrize('invalid',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_serialized_tolerance_is_rejected_before_verification(nonzero_trial_parent,invalid):
    stage=deepcopy(nonzero_trial_parent['trial_path']['candidate_stages'][0])
    stage['trials'][0]['comparison_tolerance']=invalid
    with pytest.raises(ValueError):
        revision.loads(json.dumps(stage,allow_nan=True))


def test_unrelated_equal_input_type_rules_are_unchanged():
    # The repair does not make _equal a broad NumPy-scalar alias acceptor.
    with pytest.raises(ValueError):revision._equal(np.float64(1.),1.,'Wrong external scalar type')
    with pytest.raises(ValueError):revision._equal('1',1.,'Wrong external scalar type')
    with pytest.raises(ValueError):revision._equal(1.00001,1.,'Changed tolerance ratio')
    revision._equal(1.,1.,'Ordinary JSON float')
