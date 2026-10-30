from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_dura_efficiency import r17_native_cache as cache


def test_only_original_final_third_stage_is_rendered():
    x=torch.ones(2,2,256)
    class Teacher:
        def stage_outputs(self,received):return [received,received*2,received*3],None,None,None
    torch.testing.assert_close(cache.native_iq(Teacher(),x),x*3)
    class Wrong:
        def stage_outputs(self,received):return [received,received*2],None,None,None
    with pytest.raises(ValueError):cache.native_iq(Wrong(),x)


def test_nonfinite_or_wrong_final_shape_is_rejected():
    x=torch.ones(2,2,256)
    for final in (torch.full_like(x,float('nan')),x[:1]):
        teacher=SimpleNamespace(stage_outputs=lambda value:([value,value,final],None,None,None))
        with pytest.raises(FloatingPointError):cache.native_iq(teacher,x)


def test_source_cache_cannot_dispatch_scoring_or_target_reads():
    args=SimpleNamespace(root='root',commit='a'*40)
    assert cache.command(args,'worker','scratch_joint')[3]==cache.MODULE
    with pytest.raises(ValueError):cache.command(args,'score','scratch_joint','target')
    with pytest.raises(SystemExit):cache.parse_arguments(['score','--root','root','--commit','a'*40])


def test_fixed_twenty_view_budget_is_not_training():
    c=cache.configuration()
    assert c['source_corruption_views']==list(range(1,21))
    assert c['cache_inputs']==1260000 and c['cache_numeric_bytes']==2580480000
    assert c['teacher_component']=='native_map_prior.stage3' and c['teacher_parameters']==4665073
    assert c['new_training_epochs']==c['model_optimizer_updates']==c['F_reads']==c['TX_labels_read']==c['target_reads']==0
    assert c['allowed_GPUs']==[4] and c['max_total_visible_GPU_processes']==3
