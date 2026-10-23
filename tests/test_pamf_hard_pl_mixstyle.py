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
from experiments.pamf_ssl_research.hard_pl_mixstyle import RUN,METHOD,STYLE,configuration,validate,slot_ready
from experiments.pamf_ssl_research.hard_pl_mixstyle_publish import prepare
from experiments.pamf_ssl_research.hard_pl_mixstyle_objective import HardPLMixStyleObjective,style_hooks
from tests.test_pamf_hard_pl_release import fixture


class Student(nn.Module):
    def __init__(self):
        super().__init__();self.id_backbone=nn.Module();self.id_backbone.encoder=nn.Module()
        self.id_backbone.encoder.core=nn.Module();core=nn.Module();self.id_backbone.encoder.core.id_backbone=core
        core.time_down=nn.Conv1d(2,4,1);core.t2=nn.Module();core.t2.pool=nn.Conv1d(4,4,1);core.cls_head=nn.Linear(4,6)
        self._cd_U_seen=self._cd_U_accepted=self.student_calls=self.teacher_calls=0
        self.teacher=torch.tensor([[10.,0,0,0,0,0],[10.,0,0,0,0,0],[0.,10,0,0,0,0],
                                   [0.,10,0,0,0,0],[0.,0,0,0,0,0],[0.,0,0,0,0,0]])

    def forward(self,x):
        assert not self.training and all(not m._forward_hooks for m in self.modules())
        self.teacher_calls+=1;return self.teacher

    def forward_identity_only(self,x,domain_labels=None):
        self.student_calls+=1;core=self.id_backbone.encoder.core.id_backbone
        h=core.t2.pool(core.time_down(x));self.last_logits=core.cls_head(h.mean(-1))
        return dict(tx_logits=self.last_logits)


def objective_batch(model):
    x=torch.randn(6,2,8);batch=(x,torch.full((6,),-1),torch.zeros(6,dtype=torch.long),dict(rx_i=torch.tensor([1,3]*3)))
    model._cd_U_loader=[batch];model._cd_U_iter=iter([batch])
    c=dict(hard_pl=dict(method='HPL2',threshold=.95,weight=2.),hard_pl_label_smoothing=.01,
           hard_pl_mixstyle=deepcopy(STYLE),source_rxs=[1,3],hard_pl_smoke=True)
    return HardPLMixStyleObjective(c)


def test_actual_style_only_accepted_student_once_with_masked_smoothed_loss():
    torch.manual_seed(39);model=Student().train();objective=objective_batch(model)
    loss=objective.loss(model,'cpu');mask=torch.tensor([1.,1.,1.,1.,0.,0.])
    expected=2*(F.cross_entropy(model.last_logits,model.teacher.argmax(1),reduction='none',label_smoothing=.01)*mask).mean()
    torch.testing.assert_close(loss,expected);loss.backward()
    assert model.teacher_calls==model.student_calls==1 and model.training
    state=objective.state_dict();style=state['U_style']
    assert style['seen']==6 and style['eligible']==4 and set(style['applied'].values())=={4}
    assert state['accepted']==4 and state['physical_seen']==state['EMA_updates']==0
    for h in objective.style_smoke_maps.values():
        assert torch.isfinite(h.grad).all() and h.grad[:4].norm()>0 and h.grad[4:].norm()==0
    assert not objective.style_active and all(not m._forward_hooks for m in model.modules())


def test_all_rejected_has_zero_gradient_and_no_style_partner():
    model=Student().train();objective=objective_batch(model);model.teacher.zero_()
    loss=objective.loss(model,'cpu');loss.backward();state=objective.state_dict()
    assert loss==0 and state['U_style']['eligible']==0 and set(state['U_style']['applied'].values())=={0}
    assert all(p.grad is None or not p.grad.count_nonzero() for p in model.parameters())


def test_foreign_RX_rejected_and_exception_removes_all_style_hooks():
    model=Student().train();objective=objective_batch(model)
    model._cd_U_loader[0][3]['rx_i'][0]=99
    with pytest.raises(ValueError,match='source RX'):objective.loss(model,'cpu')
    assert all(not m._forward_hooks for m in model.modules())
    core=model.id_backbone.encoder.core.id_backbone
    recipe=dict(STYLE,levels=['time_down','t2.pool'])
    with pytest.raises(RuntimeError,match='probe'):
        with style_hooks(core,'',recipe,torch.arange(6),torch.zeros(6,dtype=torch.bool),torch.ones(6,1,1)):
            raise RuntimeError('probe')
    assert all(not m._forward_hooks for m in model.modules())


def test_profile_rejects_query_truth_unknown_recipe_and_old_profiles():
    from experiments.pamf_ssl_research.hard_pl_ls import validate as old_validate
    c=configuration(fixture()['2026092701']['F21']['train'],'/p','new')
    assert c['hard_pl_label_smoothing']==.01 and c['hard_pl']['weight']==2 and c['hard_pl']['threshold']==.95
    with pytest.raises(ValueError):old_validate(c)
    for change in ({'target_inputs':'x'},{'hard_pl_no_uas':True},{'own_EMA':True},{'hard_pl_mixstyle':dict(STYLE,alpha=.2)}):
        b=deepcopy(c);b.update(change)
        with pytest.raises(ValueError):validate(b)


def test_complete_three_seeds_both_readouts_and_single_prior_scorer():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==[METHOD] and len(rt['rows'])==3 and rt['max_total_workers']==3
    assert rt['prior_score_owner_state']==rt['LS_GPU_owner_state']
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert c['hard_pl_mixstyle']==STYLE and c['output_root']==p['source_output']
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']


def test_GPU_handoff_reserves_healthy_LS_before_CUDA_and_never_third_task():
    from experiments.pamf_ssl_research.control import pending_gpu_pids,capacity
    seed=2026092701;rt=dict(gpu_by_seed={str(seed):5},hard_PL_GPU_owner_state='h',gpu_predecessor_state='n',LS_GPU_owner_state='l')
    states=dict(h=dict(jobs={m+'-s'+str(seed):dict(status='SCORED') for m in ('HPL2','HPLT1U')}),
                n=dict(jobs={'HPL_NO_UAS-s'+str(seed):dict(status='SCORED')}),
                l=dict(jobs={'HPL_LS2-s'+str(seed):dict(status='RUNNING',pid=99)}))
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:states[p]):
        assert slot_ready(rt,seed) and not slot_ready(rt,2026092702)
        own=[dict(gpu=5,child=SimpleNamespace(pid=101))]
        assert pending_gpu_pids(rt,(METHOD,),seed,own)=={99,101}
        with patch('experiments.pamf_ssl_research.control.subprocess.check_output',side_effect=['12\n13\n','19000']):
            assert capacity(5,pending_gpu_pids(rt,(METHOD,),seed))==(3,19000)
        states['h']['jobs']['HPLT1U-s'+str(seed)]['status']='RUNNING'
        assert not slot_ready(rt,seed)


def test_real_checkpoint_smoke_precedes_any_formal_dispatch():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_mixstyle_worker' in command
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_mixstyle.slot_ready',return_value=True), \
             patch('experiments.pamf_ssl_research.hard_pl_mixstyle.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,19000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            launch.assert_not_called()


def test_full_report_requires_complete_fixed_matrix_and_keeps_raw_goal():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    from tests.test_pamf_ssl_goal_counts import save
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=[METHOD])
        rt['rows']=[r for r in rt['rows'] if r['method']=='PSM']
        for row in rt['rows']:
            row.update(method=METHOD,row_id=METHOD+'-s'+str(row['model_seed']))
            c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id']);save(Path(row['score_config']),c)
            path=Path(c['output_root'])/'scores.json';v=json.loads(path.read_text());v.update(run_id=RUN,row_id=row['row_id']);save(path,v)
        report(rt,root/'report');analysis=json.loads((root/'report/analysis.json').read_text())
        assert {v['method'] for v in analysis['verdicts']}=={METHOD} and not analysis['goal_met']
