from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import MethodType,SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from experiments.pamf_ssl_research.hard_pl_guard import RUN,METHODS,GUARD,configuration,validate,slot_ready
from experiments.pamf_ssl_research.hard_pl_guard_publish import prepare
from experiments.pamf_ssl_research.hard_pl_guard_objective import GuardedHardPL,GuardedNoAsMixStyle,objective
from experiments.pamf_ssl_research.hard_pl_early_train import aggregate
from tests.test_pamf_hard_pl_early import state
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


@pytest.mark.parametrize('method',METHODS)
def test_fixed_source_only_guarded_recipe_and_negative_permissions(method):
    c=configuration(fixture()['2026092701']['F21']['train'],method,'/p','new')
    assert c['hard_pl']['weight']==6 and c['hard_pl']['threshold']==.95 and c.get('hard_pl_label_smoothing',.01)==.01
    assert c['hard_pl_parent_epoch']==20 and c['hard_pl_continuation_epochs']==30 and c['optimizer_updates']==11100
    assert isinstance(objective(c),GuardedHardPL if method==METHODS[0] else GuardedNoAsMixStyle)
    for change in ({'target_inputs':'x'},{'posthoc_head_fit':True},{'hard_pl_parent_epoch':30},
        {'hard_pl':dict(c['hard_pl'],weight=8)},{'hard_pl_gradient_guard':dict(GUARD,U_to_L_norm_cap=2)}):
        bad=deepcopy(c);bad.update(change)
        with pytest.raises(ValueError):validate(bad)


def test_old_method_cannot_silently_enable_new_gradient_guard():
    from experiments.pamf_ssl_research.hard_pl_early import configuration as old,validate as validate_old,METHODS as OLD
    c=old(fixture()['2026092701']['F21']['train'],OLD[0],'/p','new');c['hard_pl_gradient_guard']=deepcopy(GUARD)
    with pytest.raises(ValueError):validate_old(c)


def test_complete_supervised_anchor_is_preserved_by_actual_hook_backward():
    from experiments.pamf_ssl_research.hard_pl_train import installed
    from experiments.cric_dura_comparison import t1_model
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.theta=torch.nn.Parameter(torch.ones(3))
            def hook(self,**kwargs):
                return 2*self.theta[1]+t1_model.plain_pl(self,'cpu'),{}
            self._source_labeled_feature_hook=MethodType(hook,self)
    class Controlled(GuardedHardPL):
        def loss(self,student,device):
            self.calls+=1;self.pending_gradient_loss=6*(-2/3*student.theta[0]-student.theta[1]+.5*student.theta[2])
            return self.pending_gradient_loss.detach()*0.
    c=configuration(fixture()['2026092701']['F21']['train'],METHODS[0],'/p','new');holder={};model=Toy()
    with patch.object(t1_model,'install',side_effect=lambda model,c:model):
        with installed(c,holder,objective_cls=Controlled):
            model=t1_model.install(model,c);ce=model.theta[0]
            correction,metrics=model._source_labeled_feature_hook(identity_loss=ce)
            (ce+correction).backward()
    gl=torch.tensor([1.,2.,0.]);projected=torch.tensor([-.8,.4,3.]);applied=projected*(5/9.8)**.5
    assert torch.allclose(model.theta.grad,gl+applied,atol=1e-6)
    assert torch.dot(gl,model.theta.grad)>=torch.dot(gl,gl)-1e-6
    assert (model.theta.grad-gl).norm()<=gl.norm()+1e-6 and model.theta.grad[2]>0
    obj=holder['objective'];guard=obj.state_dict()['U_gradient_guard']
    assert guard['calls']==1 and not guard['pending_U_loss'] and guard['conflicts']['other']==guard['caps']['other']==1
    with pytest.raises(ValueError):obj.guide(ce,model)


@pytest.mark.parametrize('mechanisms',[False,True])
def test_guard_state_closes_both_native_segments_with_correct_weight(mechanisms):
    with tempfile.TemporaryDirectory() as tmp:
        parent=Path(tmp);save(parent/'progress.json',dict(PL_seen=396900,PL_accepted=390000))
        states=[state(e,mechanisms) for e in (10,20)]
        for s in states:s['U_gradient_guard']=dict(recipe=deepcopy(GUARD),calls=s['calls'],pending_U_loss=False,
            conflicts={g:s['calls']//2 for g in GUARD['groups']},caps={g:2 for g in GUARD['groups']})
        a=aggregate(states,[dict(folder=str(parent))],weight=6.)
        assert a['weight']==6 and a['U_gradient_guard']['calls']==6660 and a['U_gradient_guard']['caps']['t2']==4
        assert a['student_U_views']==(10602900 if mechanisms else 2097900)
        assert a['hard_PL_tail']['U_gradient_guard']==a['U_gradient_guard']
        states[-1]['U_gradient_guard']['pending_U_loss']=True
        with pytest.raises(ValueError,match='consume'):aggregate(states,[dict(folder=str(parent))],weight=6.)


def test_six_rows_keep_same_parent_native_budget_and_both_readouts():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==list(METHODS) and len(rt['rows'])==6 and rt['max_total_workers']==6
    assert rt['prior_score_owner_state']==rt['early_GPU_owner_state']
    for row in rt['rows']:
        c,p,s=(blobs['configs/'+row['row_id']+'-'+k+'.json'] for k in ('train','predict','score'))
        assert validate(c) is c and c['run_id']==p['source_run_id']==RUN and p['source_output']==c['output_root']
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']


def test_handoff_preserves_both_old_starting_workers_and_reserves_own_PID():
    from experiments.pamf_ssl_research.control import pending_gpu_pids
    from experiments.pamf_ssl_research.hard_pl_early import METHODS as OLD
    seed=2026092701;rt=dict(gpu_by_seed={str(seed):5},early_GPU_owner_state='old')
    old={m+'-s'+str(seed):dict(status='RUNNING',pid=99+i) for i,m in enumerate(OLD)};jobs={}
    with patch('experiments.cric_dura_comparison.io.read',return_value=dict(jobs=old)):
        assert not slot_ready(rt,seed,METHODS[0],jobs)
        assert pending_gpu_pids(rt,METHODS,seed,[dict(gpu=5,child=SimpleNamespace(pid=101))])=={99,100,101}
        old[OLD[0]+'-s'+str(seed)]['status']='SCORED'
        assert slot_ready(rt,seed,METHODS[0],jobs) and not slot_ready(rt,seed,METHODS[1],jobs)
        old[OLD[1]+'-s'+str(seed)]['status']='PREDICTIONS_COMPLETE'
        assert not slot_ready(rt,seed,METHODS[1],jobs)
        jobs[METHODS[0]+'-s'+str(seed)]=dict(status='RUNNING',pid=101)
        assert slot_ready(rt,seed,METHODS[1],jobs)


def test_real_guard_smoke_precedes_any_formal_row_launch():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_guard_worker' in command
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_guard.slot_ready',return_value=True), \
             patch('experiments.pamf_ssl_research.hard_pl_guard.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,19000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            launch.assert_not_called()


def test_complete_guard_report_does_not_substitute_DURA_for_raw_goal():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=list(METHODS))
        rt['rows']=[r for r in rt['rows'] if r['method'] in ('PSM','RXPSM')]
        for row in rt['rows']:
            method=METHODS[0] if row['method']=='PSM' else METHODS[1];row.update(method=method,row_id=method+'-s'+str(row['model_seed']))
            c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id']);save(Path(row['score_config']),c)
            path=Path(c['output_root'])/'scores.json';v=json.loads(path.read_text());v.update(run_id=RUN,row_id=row['row_id']);save(path,v)
        report(rt,root/'report');a=json.loads((root/'report/analysis.json').read_text())
        assert len(a['verdicts'])==4 and {v['method'] for v in a['verdicts']}==set(METHODS) and not a['goal_met']
