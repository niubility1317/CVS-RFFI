from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from experiments.pamf_ssl_research.hard_pl_step_guard import RUN,METHODS,STEP_GUARD,configuration,validate,slot_ready,smoke_slot
from experiments.pamf_ssl_research.hard_pl_step_guard_publish import prepare
from experiments.pamf_ssl_research.hard_pl_branch_guard import METHODS as BG
from experiments.pamf_ssl_research.hard_pl_early_train import aggregate
from tests.test_pamf_hard_pl_branch_guard import branch_states
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def config():return configuration(fixture()['2026092701']['F21']['train'],'/p','new')


def test_fixed_recipe_and_current_moment_reference():
    c=config();assert validate(c) is c and c['hard_pl_step_guard']==STEP_GUARD
    assert c['hard_pl']['weight']==6 and c['hard_pl']['threshold']==.95
    assert c['hard_pl_parent_epoch']==20 and c['optimizer_updates']==11100
    assert c['hard_pl_branch_guard']['extra_autograd_passes']==2 and not c['posthoc_head_fit']


def test_real_objective_captures_existing_four_derivatives_without_extra_forward():
    from experiments.pamf_ssl_research.hard_pl_step_guard_objective import StepGuardedNativeAs
    from tests.test_pamf_hard_pl_native_as import batch
    from experiments.cric_dura_comparison import t1_model
    model,_,x=batch(smoke=True);c=config();c.update(hard_pl_smoke=True,source_rxs=[1,3],step_offset=4440)
    obj=StepGuardedNativeAs(c);obj.loss(model,'cpu');y=torch.tensor([0,0,1,1,2,2])
    a=model.forward_identity_only(x)['tx_logits'];b=a+.05*torch.randn_like(a)
    original=torch.autograd.grad;calls=[]
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    with obj.labeled_context(),patch.object(obj.parameter_guard,'capture') as capture:
        parts=[t1_model.pair(a,b,y,kl_weight=0. if i==2 else 1.)[0] for i in range(4)]
        labeled=.5*parts[0]+.25*parts[1]+.25*parts[2]+.25*parts[3]+a.square().mean()*.01
        with patch('torch.autograd.grad',side_effect=counted):value=obj.guide(labeled,model)
        value.backward();capture.assert_called_once()
        assert capture.call_args.args[0] is model and len(capture.call_args.args)==5
    assert len(calls)==4 and model.teacher_calls==1 and model.student_calls==7


@pytest.mark.parametrize('change',[{'target_inputs':'x'},{'posthoc_head_fit':True},{'hard_pl_parent_epoch':30},
    {'hard_pl_step_guard':dict(STEP_GUARD,real_joint_moments_retained=False)},
    {'hard_pl_step_guard':dict(STEP_GUARD,extra_model_forwards=1)}])
def test_invalid_data_or_update_recipe_rejected(change):
    c=config();c.update(change)
    with pytest.raises(ValueError):validate(c)


def test_release_declares_only_three_fixed_seeds_with_dual_readouts():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==list(METHODS) and len(rt['rows'])==rt['max_total_workers']==3
    assert rt['prior_score_owner_state']==rt['branch_guard_GPU_owner_state']
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert validate(c) is c and p['source_output']==c['output_root'] and p['source_run_id']==RUN
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']


def states():
    values=branch_states()
    for s in values:
        n=s['calls'];s['U_parameter_step_guard']=dict(recipe=deepcopy(STEP_GUARD),calls=n,anchor_captures=n,
            reference_optimizer_steps=n,reference_physical_clip_calls=n,pending_anchors=False,
            copied_existing_moments=True,live_joint_optimizer_state_retained=True,reference_has_no_model_forward=True,
            extra_autograd_calls=0,extra_model_forwards=0,fallback_groups=0,removed_material_conflicts=n,
            minimum_whole_increment_retention=.8,seconds=5.,U_truth_reads=0,target_reads=0,new_head=False,new_inference_parameters=0)
    return values


def test_two_segments_aggregate_exact_extra_optimizer_budget():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp);save(path/'progress.json',dict(PL_seen=396900,PL_accepted=390000));v=states()
        result=aggregate(v,[dict(folder=tmp)],weight=6.)
        assert result['U_parameter_step_guard']['calls']==6660
        assert result['U_parameter_step_guard']['reference_optimizer_steps']==6660
        assert result['student_U_views']==10602900 and result['U_branch_guard']['extra_autograd_calls']==13320
        v[-1]['U_parameter_step_guard']['pending_anchors']=True
        with pytest.raises(ValueError):aggregate(v,[dict(folder=tmp)],weight=6.)


def test_queue_reserves_matching_branch_guard_pending_cuda_worker():
    from experiments.pamf_ssl_research.control import pending_gpu_pids
    rt=dict(gpu_by_seed={'2026092701':5,'2026092702':6,'2026092703':7},branch_guard_GPU_owner_state='bg')
    source=dict(jobs={BG[0]+'-s'+str(seed):dict(status='QUEUED') for seed in (2026092701,2026092702,2026092703)})
    with patch('experiments.cric_dura_comparison.io.read',return_value=source):
        assert not slot_ready(rt,2026092702)
        source['jobs'][BG[0]+'-s2026092702'].update(status='RUNNING',pid=123)
        assert smoke_slot(rt,lambda *a:(2,18000))==(2026092702,6)
        assert pending_gpu_pids(rt,METHODS,2026092702,[dict(gpu=6,child=SimpleNamespace(pid=99))])=={123,99}
        assert smoke_slot(rt,lambda *a:(3,18000)) is None


def test_matching_real_checkpoint_smoke_is_first_gpu_action():
    from experiments.pamf_ssl_research.control import run
    class Reached(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_step_guard_worker' in command and kwargs['env']['CUDA_VISIBLE_DEVICES']=='6'
            c=json.loads(Path(command[-1]).read_text());assert c['rows'][0]['model_seed']==2026092702
            raise Reached
        with patch('experiments.pamf_ssl_research.hard_pl_step_guard.smoke_slot',return_value=(2026092702,6)), \
             patch('experiments.pamf_ssl_research.hard_pl_step_guard.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,18000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(Reached):run(rt)
            launch.assert_not_called()


def test_target_reporting_keeps_raw_goal_and_separate_DURA_reference():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=list(METHODS));rt['rows']=[r for r in rt['rows'] if r['method']=='PSM']
        for row in rt['rows']:
            row.update(method=METHODS[0],row_id=METHODS[0]+'-s'+str(row['model_seed']))
            c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id']);save(Path(row['score_config']),c)
            p=Path(c['output_root'])/'scores.json';v=json.loads(p.read_text());v.update(run_id=RUN,row_id=row['row_id']);save(p,v)
        report(rt,root/'report');d=json.loads((root/'report/analysis.json').read_text())
        assert len(d['verdicts'])==2 and {v['method'] for v in d['verdicts']}==set(METHODS) and not d['goal_met']
