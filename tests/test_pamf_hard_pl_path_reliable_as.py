from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_ssl_research.hard_pl_path_reliable_as import RUN,METHODS,configuration,validate,slot_ready,smoke_slot
from experiments.pamf_ssl_research.hard_pl_path_reliable_as_objective import PathReliableAs,GuardedPathReliableAs
from experiments.pamf_ssl_research.hard_pl_path_reliable_as_publish import prepare
from experiments.pamf_ssl_research.hard_pl_path import PATH,METHODS as PATH_METHODS,configuration as path_configuration
from experiments.pamf_ssl_research.hard_pl_path_objective import PathComplement
from experiments.pamf_ssl_research.hard_pl_reliable_as import RELIABLE_AS,METHODS as RAS_METHODS
from experiments.pamf_ssl_research.hard_pl_early_train import aggregate
from tests.test_pamf_hard_pl_reliable_as import RasStudent
from tests.test_pamf_hard_pl_early import state
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def batch(cls=PathReliableAs,smoke=True,path_only=False):
    model=RasStudent().train();ref=model.id_backbone.encoder
    class Contribution(nn.Module):
        def __init__(self,linear):super().__init__();self.linear=linear
        def forward(self,x):return .2*self.linear(x)
    ref.sequence_pool=Contribution(ref.sequence_pool);ref.response_projection=nn.Linear(4,160)
    ref.response_gain=nn.Parameter(torch.full((160,),.2))
    class Reference(type(ref)):
        def features(self,x):
            core=self.core.id_backbone;a=core.time_down(x);h=core.t2.pool(a)
            return core.projection(h.mean(-1))+self.sequence_pool(a.square().mean(-1))+self.response_gain.tanh()*self.response_projection(a.mean(-1))
        def classify_features(self,h):return self.core.id_backbone.cls_head.classify(h)
    ref.__class__=Reference
    x=torch.randn(6,2,8);values=(x,torch.full((6,),-1),torch.zeros(6,dtype=torch.long),dict(rx_i=torch.tensor([1,3]*3)))
    model._cd_U_loader=[values];model._cd_U_iter=iter([values])
    c=(path_configuration if path_only else configuration)(fixture()['2026092701']['F21']['train'],'/p','new')
    c.update(hard_pl_smoke=smoke,source_rxs=[1,3],step_offset=4440)
    return model,cls(c),x


def test_fixed_composition_preserves_both_recipes_and_forbids_permissions_or_dose_changes():
    from experiments.pamf_ssl_research.hard_pl_path import validate as pv
    from experiments.pamf_ssl_research.hard_pl_reliable_as import validate as rv
    c=configuration(fixture()['2026092701']['F21']['train'],'/p','new')
    assert c['hard_pl_path']==PATH and c['hard_pl_reliable_as']==RELIABLE_AS and not c['hard_pl_no_uas']
    assert c['hard_pl_parent_epoch']==20 and c['optimizer_updates']==11100 and c['hard_pl']['weight']==6
    for validator in (pv,rv):
        with pytest.raises(ValueError):validator(c)
    for change in ({'target_inputs':'x'},{'posthoc_head_fit':True},{'hard_pl_parent_epoch':30},{'hard_pl_no_uas':True},
                   {'hard_pl_path':dict(PATH,core_CE_coefficient=.25)},
                   {'hard_pl_reliable_as':dict(RELIABLE_AS,augmented_view_gate=.8)}):
        bad=deepcopy(c);bad.update(change)
        with pytest.raises(ValueError):validate(bad)


def test_actual_composed_loss_consumes_both_objectives_once_and_fullL_guard_sees_both():
    torch.manual_seed(81);model,obj,x=batch(GuardedPathReliableAs);params=tuple(id(p) for p in model.parameters())
    norm={k:v.clone() for k,v in model.eval_norm.state_dict().items()};obj.loss(model,'cpu');s=obj.state_dict()
    assert model.teacher_calls==2 and model.student_calls==6 and model._cd_practical.calls==1
    q=s['U_reliable_As'];p=s['U_path_complement']
    assert (q['generated'],q['used'],q['raw_fallback'],q['raw_accepted_half'])==(3,1,2,2)
    assert p['calls']==q['calls']==s['U_joint_contrast']['calls']==1 and p['extra_head_calls']==4 and p['extra_head_packet_views']==24
    assert s['accepted']==4 and s['student_views']==36 and s['physical_student_views']==2 and s['raw_As_seen']==2
    torch.testing.assert_close(model.student_inputs[2][0],x[0]+.25)
    torch.testing.assert_close(model.student_inputs[2][[2,4]],x[[2,4]])
    assert obj.smoke_path_gradients['signature']<=1e-7 and obj.smoke_path_gradients['response']<=1e-7
    assert min(obj.smoke_path_gradients['head'],obj.smoke_path_gradients['core'],obj.smoke_joint_signature_gradient)>0
    for k,v in norm.items():torch.testing.assert_close(model.eval_norm.state_dict()[k],v)
    labeled=F.cross_entropy(model.forward_identity_only(x)['tx_logits'],torch.tensor([0,0,1,1,2,2]))
    obj.guide(labeled,model).backward()
    for name,param in model.named_parameters():
        expected=obj.smoke_guard_expected[name]
        if expected is None:assert param.grad is None
        else:torch.testing.assert_close(param.grad,expected)
    assert tuple(id(p) for p in model.parameters())==params and model.training
    assert all(not m._forward_hooks for m in model.modules()) and obj._path_current is None and obj._joint_current is None


def test_all_As_rejected_reproduces_Path_objective_without_losing_accepted_raw_samples():
    torch.manual_seed(83);first,base,_=batch(PathComplement,False,True)
    torch.manual_seed(83);second,new,_=batch(PathReliableAs,False);second.As_teacher.zero_()
    torch.manual_seed(84);old_loss=base.loss(first,'cpu')
    torch.manual_seed(84);new_loss=new.loss(second,'cpu')
    torch.testing.assert_close(new_loss,old_loss)
    assert base.state_dict()['accepted']==new.state_dict()['accepted']==4
    assert new.state_dict()['U_reliable_As']['used']==0 and new.state_dict()['U_path_complement']['extra_head_calls']==4


def test_all_raw_rejected_and_extra_teacher_failure_leave_no_hidden_gradient_or_hooks():
    model,obj,_=batch(smoke=False);model.teacher.zero_();loss=obj.loss(model,'cpu');loss.backward()
    assert loss==0 and obj.state_dict()['U_reliable_As']['used']==0
    assert all(p.grad is None or not p.grad.count_nonzero() for p in model.parameters())
    model,obj,_=batch(smoke=False);model.fail_As=True
    with pytest.raises(RuntimeError,match='source As eval failure'):obj.loss(model,'cpu')
    assert model.training and not obj.ras_active and not obj.path_active and not obj.joint_active
    assert obj._path_current is None and obj._joint_current is None and all(not m._forward_hooks for m in model.modules())


def test_aggregate_records_extra_head_and_As_teacher_costs_together_without_more_student_forwards():
    with tempfile.TemporaryDirectory() as tmp:
        parent=Path(tmp);save(parent/'progress.json',dict(PL_seen=396900,PL_accepted=390000));ss=[state(e,True) for e in (10,20)]
        for s in ss:
            g=s['seen']//2;u=g//2;s.update(physical_seen=g,physical_student_views=2*u,raw_As_seen=g-u)
            s['U_reliable_As']=dict(recipe=deepcopy(RELIABLE_AS),calls=s['calls'],generated=g,teacher_As_views=g,used=u,raw_fallback=g-u,
                raw_accepted_half=g,view_confident=g,view_argmax_agree=u,all_raw_accepted_retained=True,
                extra_student_forwards=0,new_head=False,EMA_updates=0,U_truth_read=False,teacher_eval_norm_updates=0)
            s['U_path_complement']=dict(recipe=deepcopy(PATH),calls=s['calls'],extra_head_calls=4*s['calls'],extra_head_packet_views=4*s['seen'],
                extra_backbone_forwards=0,teacher_or_L_head_calls=0,new_head=False,persistent_hooks=False,
                full_deployment_signature_retained=True,full_deployment_response_retained=True,CE_dose_normalizer=1.25)
        result=aggregate(ss,[dict(folder=str(parent))],weight=6.)
        assert result['teacher_eval_U_views']==2948400 and result['student_U_views']==10602900
        assert result['U_path_complement']['extra_head_calls']==26640 and result['U_path_complement']['extra_head_packet_views']==6804000
        assert result['U_reliable_As']['generated']==850500 and result['physical_student_U_views']==850500


def test_release_keeps_defaulthead_three_seeds_all_readouts_and_single_scorer_after_ReliableAs():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==list(METHODS) and rt['prior_score_owner_state']==rt['reliable_as_GPU_owner_state']
    assert len(rt['rows'])==3 and rt['max_total_workers']==3
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert validate(c) is c and p['source_output']==c['output_root']
        assert p['source_run_id']==RUN and list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']


def test_slot_waits_for_finished_Path_GPU_predictions_and_dispatched_ReliableAs():
    from experiments.pamf_ssl_research.control import pending_gpu_pids
    rt=dict(gpu_by_seed={'2026092701':5,'2026092702':6,'2026092703':7},path_GPU_owner_state='p',reliable_as_GPU_owner_state='r')
    states={k:dict(jobs={}) for k in ('p','r')}
    for seed in (2026092701,2026092702,2026092703):
        states['p']['jobs'][PATH_METHODS[0]+'-s'+str(seed)]=dict(status='RUNNING',pid=seed)
        states['r']['jobs'][RAS_METHODS[0]+'-s'+str(seed)]=dict(status='QUEUED')
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:states[p]):
        assert not slot_ready(rt,2026092702)
        states['p']['jobs'][PATH_METHODS[0]+'-s2026092702']['status']='PREDICTIONS_COMPLETE'
        assert not slot_ready(rt,2026092702)
        states['r']['jobs'][RAS_METHODS[0]+'-s2026092702'].update(status='RUNNING',pid=123)
        assert slot_ready(rt,2026092702) and smoke_slot(rt,lambda *a:(2,18000))==(2026092702,6)
        assert pending_gpu_pids(rt,METHODS,2026092702,[dict(gpu=6,child=SimpleNamespace(pid=99))])=={123,99}
        assert smoke_slot(rt,lambda *a:(3,18000)) is None


def test_first_released_matching_seed_smoke_precedes_any_formal_launch():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_path_reliable_as_worker' in command and kwargs['env']['CUDA_VISIBLE_DEVICES']=='6'
            c=json.loads(Path(command[-1]).read_text());assert c['rows'][0]['model_seed']==2026092702
            assert {r['model_seed']:r['gpu'] for r in c['rows']}=={2026092701:5,2026092702:6,2026092703:7}
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_path_reliable_as.smoke_slot',return_value=(2026092702,6)), \
             patch('experiments.pamf_ssl_research.hard_pl_path_reliable_as.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,18000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            launch.assert_not_called()


def test_complete_report_keeps_both_reference_baselines_and_raw_goal_distinct_from_DURA():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=list(METHODS));rt['rows']=[r for r in rt['rows'] if r['method']=='PSM']
        for row in rt['rows']:
            row.update(method=METHODS[0],row_id=METHODS[0]+'-s'+str(row['model_seed']))
            c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id']);save(Path(row['score_config']),c)
            p=Path(c['output_root'])/'scores.json';v=json.loads(p.read_text());v.update(run_id=RUN,row_id=row['row_id']);save(p,v)
        report(rt,root/'report');result=json.loads((root/'report/analysis.json').read_text())
        assert len(result['verdicts'])==2 and {v['method'] for v in result['verdicts']}==set(METHODS) and not result['goal_met']
