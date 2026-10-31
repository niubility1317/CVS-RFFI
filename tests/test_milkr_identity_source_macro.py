import json
import pytest
from experiments.pamf_phase2_practical.milkr_identity_source import assess
from experiments.pamf_phase2_practical.milkr_identity_reuse_source import macro_assessment,load_source_grid,SOURCE_KEYS


def rows(improved=False):
    out=[]
    for k in (1,5,20):
        for sp in (0,1):
            for qp in (0,1):
                old,new=(.6,.9) if not improved else ((.78,.94) if qp else (.70,.92))
                if improved and (k,sp,qp)==(1,0,0):old,new=.75,.895
                out.append(dict(k=k,S=sp,Q=qp,H=2*old*new/(old+new),old_accuracy=old,new_accuracy=new))
    return out


def test_macro_objective_does_not_require_every_component_to_improve():
    control=rows();candidate=rows(True)
    assert assess(candidate,control)['eligible'] is False
    result=macro_assessment(candidate,control)
    assert result['eligible'] is True and result['previous_component_gate_pass'] is False
    assert result['min_K_Q0_gain_pp']>.25


def test_grid_reuse_rejects_target_feedback_and_contract_mismatch(tmp_path):
    parent={key:'same-'+key for key in SOURCE_KEYS};parent.update(kernel_method='identity_holdout',run_root=str(tmp_path/'parent'),run_id='source_parent')
    (tmp_path/'parent').mkdir();path=tmp_path/'runtime.json';path.write_text(json.dumps(parent))
    selected=dict(status='SOURCE_FROZEN',target_eligible=False,target_reads=0,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0,source_candidates=1)
    selection=tmp_path/'parent/source_selection.json';selection.write_text(json.dumps(selected));(tmp_path/'parent/source_calibration_grid.json').write_text(json.dumps(dict(source_only=True,target_reads=0,candidates=[{}])))
    rt=dict(parent,source_identity_grid_runtime=str(path))
    assert len(load_source_grid(rt)[2])==1
    with pytest.raises(ValueError,match='ancestry'):load_source_grid(dict(rt,source_contract='wrong'))
    selection.write_text(json.dumps(dict(selected,query_reads=1)))
    with pytest.raises(ValueError,match='disallowed'):load_source_grid(rt)
