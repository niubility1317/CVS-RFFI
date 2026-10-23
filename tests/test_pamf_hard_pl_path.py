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
from experiments.pamf_ssl_research.hard_pl_path import RUN,METHODS,PATH,configuration,validate,slot_ready,smoke_slot
from experiments.pamf_ssl_research.hard_pl_path_objective import PathComplement,GuardedPathComplement
from experiments.pamf_ssl_research.hard_pl_path_publish import prepare
from experiments.pamf_ssl_research.hard_pl_joint import METHODS as JOINT,JOINT as JOINT_RECIPE
from experiments.pamf_ssl_research.hard_pl_complement import METHODS as COMP
from experiments.pamf_ssl_research.hard_pl_guard import GUARD
from experiments.pamf_ssl_research.hard_pl_early_train import aggregate
from experiments.pamf_ssl_research.hard_pl_joint_objective import JointNoAsMixStyle
from tests.test_pamf_hard_pl_joint import Student
from tests.test_pamf_hard_pl_early import state
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def test_path_permissions_parent_head_and_conserved_CE_dose():
    from experiments.pamf_ssl_research.hard_pl_joint import validate as old_validate
    c=configuration(fixture()['2026092701']['F21']['train'],'/p','new')
    assert c['hard_pl_parent_epoch']==20 and c['hard_pl_continuation_epochs']==30 and c['optimizer_updates']==11100
    assert c['hard_pl']['weight']==6 and c['hard_pl']['threshold']==.95 and c['hard_pl_joint_contrast']==JOINT_RECIPE
    assert sum(PATH[k] for k in ('ordinary_CE_coefficient','no_signature_CE_coefficient','core_CE_coefficient'))==PATH['ordinary_CE_coefficient_before']
    with pytest.raises(ValueError):old_validate(c)
    for change in ({'target_inputs':'x'},{'posthoc_head_fit':True},{'hard_pl_parent_epoch':30},
                   {'hard_pl_path':dict(PATH,core_CE_coefficient=.25)},{'hard_pl_path':dict(PATH,new_head=True)}):
        bad=deepcopy(c);bad.update(change)
        with pytest.raises(ValueError):validate(bad)


def objective_batch(cls,smoke=True,zero_branches=False):
    model=Student().train();ref=model.id_backbone.encoder
    class NativeContribution(nn.Module):
        def __init__(self,linear):super().__init__();self.linear=linear
        def forward(self,x):return .2*self.linear(x)
    ref.sequence_pool=NativeContribution(ref.sequence_pool);ref.response_projection=nn.Linear(4,160)
    ref.response_gain=nn.Parameter(torch.full((160,),.2))
    class NativeReference(type(ref)):
        def features(self,x):
            core=self.core.id_backbone;a=core.time_down(x);h=core.t2.pool(a)
            return (core.projection(h.mean(-1))+self.sequence_pool(a.square().mean(-1))
                +self.response_gain.tanh()*self.response_projection(a.mean(-1)))
        def classify_features(self,h):return self.core.id_backbone.cls_head.classify(h)
    ref.__class__=NativeReference
    if zero_branches:
        with torch.no_grad():
            for module in (ref.sequence_pool,ref.response_projection):
                for param in module.parameters():param.zero_()
    x=torch.randn(6,2,8);batch=(x,torch.full((6,),-1),torch.zeros(6,dtype=torch.long),dict(rx_i=torch.tensor([1,3]*3)))
    model._cd_U_loader=[batch];model._cd_U_iter=iter([batch])
    c=configuration(fixture()['2026092701']['F21']['train'],'/p','new');c.update(hard_pl_smoke=smoke,source_rxs=[1,3],step_offset=4440)
    return model,cls(c),x


def test_core_CE_bypasses_two_exclusive_branches_but_trains_actual_core_head_and_fullL_guard():
    torch.manual_seed(61);model,obj,x=objective_batch(GuardedPathComplement);head=model.id_backbone.encoder.core.id_backbone.cls_head
    before=tuple(id(p) for p in model.parameters());obj.loss(model,'cpu')
    assert model.teacher_calls==1 and model.student_calls==6 and tuple(id(p) for p in model.parameters())==before
    g=obj.smoke_path_gradients
    assert max(g['signature'],g['response'])<=1e-7 and min(g['head'],g['core'])>0
    assert obj.smoke_joint_signature_gradient>0 and model.id_backbone.encoder.core.id_backbone.cls_head is head
    labeled=F.cross_entropy(model.forward_identity_only(x)['tx_logits'],torch.tensor([0,0,1,1,2,2]))
    obj.guide(labeled,model).backward()
    for name,param in model.named_parameters():torch.testing.assert_close(param.grad,obj.smoke_guard_expected[name])
    status=obj.state_dict();paths=status['U_path_complement']
    assert paths['calls']==1 and paths['extra_head_calls']==4 and paths['extra_head_packet_views']==24 and paths['CE_dose_normalizer']==1.25
    assert status['student_views']==36 and status['U_style']['calls']==6 and status['U_joint_contrast']['captures']==2
    assert paths['full_deployment_signature_retained'] and paths['full_deployment_response_retained']
    assert all(h.grad is not None and torch.isfinite(h.grad).all() and h.grad.norm()>0 for h in obj.smoke_path_features)
    assert all(not m._forward_hooks for m in model.modules()) and not obj.path_active


def test_equal_feature_paths_keep_old_loss_value_instead_of_adding_CE_dose():
    torch.manual_seed(63);first,base,_=objective_batch(JointNoAsMixStyle,smoke=False,zero_branches=True)
    torch.manual_seed(63);second,paths,_=objective_batch(PathComplement,smoke=False,zero_branches=True)
    torch.manual_seed(64);old=base.loss(first,'cpu')
    torch.manual_seed(64);new=paths.loss(second,'cpu')
    torch.testing.assert_close(new,old)
    torch.testing.assert_close(paths.metrics['HPL_path_core_CE'],paths.metrics['HPL_path_ordinary_CE'])
    torch.testing.assert_close(paths.metrics['HPL_path_no_signature_CE'],paths.metrics['HPL_path_ordinary_CE'])
    assert first.student_calls==second.student_calls==6 and paths.state_dict()['U_path_complement']['extra_head_calls']==4


def test_all_rejected_path_CE_has_no_U_parameter_gradient_and_exception_cleans_hooks():
    model,obj,_=objective_batch(PathComplement,smoke=False);model.teacher.zero_();loss=obj.loss(model,'cpu');loss.backward()
    assert loss==0 and torch.isfinite(loss) and obj.state_dict()['accepted']==0
    assert all(p.grad is None or not p.grad.count_nonzero() for p in model.parameters())
    model,obj,_=objective_batch(PathComplement,smoke=False);ref=model.id_backbone.encoder
    with patch.object(ref.response_projection,'forward',side_effect=RuntimeError('source failure')):
        with pytest.raises(RuntimeError,match='source failure'):obj.loss(model,'cpu')
    assert all(not m._forward_hooks for m in model.modules()) and not obj.path_active
    assert obj._path_current is None and 'features' not in ref.__dict__


def test_path_aggregation_counts_existing_head_views_without_more_backbone_or_removed_deployment_branch():
    with tempfile.TemporaryDirectory() as tmp:
        parent=Path(tmp);save(parent/'progress.json',dict(PL_seen=396900,PL_accepted=390000));values=[state(e,True) for e in (10,20)]
        for s in values:
            s['U_gradient_guard']=dict(recipe=deepcopy(GUARD),calls=s['calls'],pending_U_loss=False,
                conflicts={g:1 for g in GUARD['groups']},caps={g:1 for g in GUARD['groups']})
            s['U_joint_contrast']=dict(recipe=deepcopy(JOINT_RECIPE),calls=s['calls'],captures=2*s['calls'],captured_packets=2*s['seen'],
                anchors=2*s['accepted'],extra_forward_calls=0,teacher_or_L_captures=0,persistent_feature_wrapper=False)
            s['U_path_complement']=dict(recipe=deepcopy(PATH),calls=s['calls'],extra_head_calls=4*s['calls'],extra_head_packet_views=4*s['seen'],
                extra_backbone_forwards=0,teacher_or_L_head_calls=0,new_head=False,persistent_hooks=False,
                full_deployment_signature_retained=True,full_deployment_response_retained=True,CE_dose_normalizer=1.25)
        result=aggregate(values,[dict(folder=str(parent))],weight=6.)
        assert result['U_path_complement']['extra_head_calls']==26640 and result['U_path_complement']['extra_head_packet_views']==6804000
        assert result['student_U_views']==10602900 and result['hard_PL_tail']['U_path_complement']==result['U_path_complement']
        values[-1]['U_path_complement']['full_deployment_response_retained']=False
        with pytest.raises(ValueError,match='same dose'):aggregate(values,[dict(folder=str(parent))],weight=6.)


def test_path_release_preserves_fixed_seeds_both_seven_view_readouts_and_single_chained_scorer():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==list(METHODS) and len(rt['rows'])==rt['max_total_workers']==3
    assert rt['prior_score_owner_state']==rt['complement_GPU_owner_state']
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert validate(c) is c and p['source_output']==c['output_root'] and p['source_run_id']==RUN
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']


def test_path_slot_waits_for_joint_predictions_and_reserves_starting_complement_peer():
    from experiments.pamf_ssl_research.control import pending_gpu_pids
    rt=dict(gpu_by_seed={'2026092701':5,'2026092702':6,'2026092703':7},joint_GPU_owner_state='j',complement_GPU_owner_state='c')
    states={k:dict(jobs={}) for k in ('j','c')}
    for seed in (2026092701,2026092702,2026092703):
        states['j']['jobs'][JOINT[0]+'-s'+str(seed)]=dict(status='RUNNING',pid=seed)
        states['c']['jobs'][COMP[0]+'-s'+str(seed)]=dict(status='RUNNING',pid=seed+10)
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:states[p]):
        assert not slot_ready(rt,2026092701)
        assert pending_gpu_pids(rt,METHODS,2026092701,[dict(gpu=5,child=SimpleNamespace(pid=99))])=={99,2026092701,2026092711}
        states['j']['jobs'][JOINT[0]+'-s2026092702']['status']='PREDICTIONS_COMPLETE'
        assert slot_ready(rt,2026092702)
        assert smoke_slot(rt,lambda gpu,pending:(2,18000))==(2026092702,6)
        assert smoke_slot(rt,lambda *args:(3,18000)) is None


def test_path_smoke_first_free_matching_seed_GPU_precedes_formal_launch():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_path_worker' in command and kwargs['env']['CUDA_VISIBLE_DEVICES']=='6'
            c=json.loads(Path(command[-1]).read_text());assert c['rows'][0]['model_seed']==2026092702
            assert {r['model_seed']:r['gpu'] for r in c['rows']}=={2026092701:5,2026092702:6,2026092703:7}
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_path.smoke_slot',return_value=(2026092702,6)), \
             patch('experiments.pamf_ssl_research.hard_pl_path.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,18000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            launch.assert_not_called()


def test_path_report_keeps_both_matched_baselines_and_raw_goal_separate_from_DURA():
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
