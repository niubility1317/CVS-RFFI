import numpy as np
import pytest

from experiments.cric_dura_comparison.io import write
from experiments.pamf_lora_main import w24_joint_pamf as joint
from experiments.pamf_lora_main import w24_native_paired_full as evaluator


def test_paired_full_receipts_fix_both_models_before_scoring():
    for case in joint.FULL_CASES:
        base = joint.formal_receipt(case, evaluator.native.PAIR_RUN)
        expected = evaluator.formal_receipt(case)
        assert expected['modes'] == [*base['modes'][:-1], evaluator.CONTROL, evaluator.CANDIDATE]
        assert expected['receivers'] == base['receivers']
        assert expected['views'] == base['views']
        assert expected['fit_commit'] == evaluator.FIT_COMMIT
        assert expected['target_truth_read'] is False


def test_matched_full_uses_completion_not_source_accuracy(tmp_path):
    fit_root = tmp_path/'fit'
    release = tmp_path/'release'
    release.mkdir()
    write(release/'submit.json', {'pid':2147483647})
    for arm in evaluator.native.PAIR_ARMS:
        out = fit_root/arm
        out.mkdir(parents=True)
        write(out/'complete.json', {'status':'SOURCE_FROZEN'})
    evaluator.await_original_dispatch(fit_root,release)
    (fit_root/evaluator.CANDIDATE/'complete.json').unlink()
    with pytest.raises(ValueError,match='before both source weights froze'):
        evaluator.await_original_dispatch(fit_root,release)


def test_reuse_reorders_frozen_candidate_without_changing_base_or_truth(tmp_path):
    case = joint.FULL_CASES[0]
    previous = tmp_path/'predictions'/case
    previous.mkdir(parents=True)
    write(previous/'complete.json',joint.formal_receipt(case,evaluator.native.PAIR_RUN))
    assert evaluator.baseline_for({},tmp_path,case) == (previous,(evaluator.CONTROL,))
    value = np.arange(12,dtype=np.int16).reshape(1,4,3)
    assert np.array_equal(evaluator.order_pair(value,[1,2,3],reused=False)[0],value)
    ordered, counts = evaluator.order_pair(value,[1,2,3],reused=True)
    assert np.array_equal(ordered,value[:,[0,1,3,2]])
    assert counts == [1,3,2]
