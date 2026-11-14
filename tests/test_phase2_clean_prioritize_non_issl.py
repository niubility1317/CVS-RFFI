import json
import pytest

from experiments.pamf_phase2_practical.clean_prioritize_non_issl import prepare


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')
    return path


def test_refuses_to_repeat_a_non_issl_task_with_partial_output(tmp_path):
    old=tmp_path/'old';control=old/'scheduler'
    put(control/'state.json',dict(pending=0,active=[],failed=[]))
    put(control/'owner.json',dict(pid=0))
    put(control/'USER_STOP.json',{})
    orth=tmp_path/'orth'
    for seed in range(2026092701,2026092705):
        d=orth/'source'/f'orthogonal-s{seed}'
        put(d/'group1_complete.json',dict(status='GROUP1_SOURCE_READY'))
        put(d/'recipe.json',dict(schema='group1_source_orthogonal_tempered_PL_v2'))
        (d/'source.pt').write_bytes(b'x')
    output=old/'tasks/previous';output.mkdir(parents=True)
    job=put(tmp_path/'job.json',dict(profile='protonet_cda',seed=2026092701,capsule='capsules/P01/MIX01/1-1'))
    config=put(tmp_path/'config.json',dict(run_root=str(old),jobs=[dict(id='previous',group='native_da',
                                                    job=str(job),completion=str(output/'complete.json'))]))
    with pytest.raises(ValueError,match='prior output'):
        prepare(config,tmp_path/'plan',tmp_path/'new',tmp_path/'release',orth)
    assert not (tmp_path/'plan').exists()
