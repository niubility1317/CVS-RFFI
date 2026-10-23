from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from experiments.pamf_ssl_research.hard_pl_no_as_mixstyle import RUN,METHOD,STYLE,configuration,validate,slot_ready
from experiments.pamf_ssl_research.hard_pl_no_as_mixstyle_publish import prepare
from experiments.pamf_ssl_research.hard_pl_no_as_mixstyle_objective import NoAsMixStyleObjective
from tests.test_pamf_hard_pl_mixstyle import Student
from tests.test_pamf_hard_pl_release import fixture


def test_six_U_views_share_one_plan_and_actual_contrast_captures_post_style():
    class RecordedStudent(Student):
        def forward_identity_only(self,x,domain_labels=None):
            self.student_calls+=1;core=self.id_backbone.encoder.core.id_backbone
            a=core.time_down(x);b=core.t2.pool(a);self.maps.append(dict(time_down=a,**{'t2.pool':b}))
            return dict(tx_logits=core.cls_head(b.mean(-1)))
    torch.manual_seed(47);model=RecordedStudent().train();model.maps=[];model._feature_l_steps=1
    model._cd_practical=lambda *args:(_ for _ in ()).throw(AssertionError('U physical generator must not execute'))
    batch=(torch.randn(6,2,8),torch.full((6,),-1),torch.zeros(6,dtype=torch.long),dict(rx_i=torch.tensor([1,3]*3)))
    model._cd_U_loader=[batch];model._cd_U_iter=iter([batch])
    c=dict(hard_pl=dict(method='HPLT1U',threshold=.95,weight=2.),hard_pl_no_uas=True,
           hard_pl_mixstyle=deepcopy(STYLE),hard_pl_shared_style_plan=True,source_rxs=[1,3],hard_pl_smoke=True,step_offset=6660)
    objective=NoAsMixStyleObjective(c);plans=[];captures=[]
    from experiments.pamf_ssl_research.u_mixstyle_objective import mix_moments
    def mix(h,partner,valid,lam,eps):
        plans.append((partner.detach().clone(),lam.detach().clone()));return mix_moments(h,partner,valid,lam,eps)
    def contrast(maps,y,rx):
        captures.append({k:v.detach().clone() for k,v in maps.items()})
        return sum(v.square().mean() for v in maps.values()),[dict(level=k,valid_anchors=len(y)) for k in maps]
    with patch('experiments.pamf_ssl_research.hard_pl_mixstyle_objective.mix_moments',side_effect=mix), \
         patch('experiments.pamf_ssl_research.hard_pl_objective.contrast',side_effect=contrast):
        loss=objective.loss(model,'cpu');loss.backward()
    state=objective.state_dict();style=state['U_style']
    assert model.teacher_calls==1 and model.student_calls==6 and len(plans)==12
    assert all(torch.equal(p,plans[0][0]) and torch.equal(lam,plans[0][1]) for p,lam in plans)
    assert style['calls']==6 and style['seen']==36 and style['eligible']==24 and style['plan_calls']==1
    assert set(style['applied'].values())=={24} and style['shared_six_forward_plan']
    for i,captured in enumerate(captures):
        for name,h in captured.items():torch.testing.assert_close(h,model.maps[i][name][:4])
    assert state['student_views']==36 and state['raw_As_seen']==3 and state['physical_seen']==0
    assert torch.isfinite(loss) and all(not m._forward_hooks for m in model.modules()) and not objective.style_active


def test_only_style_factor_changes_from_fixed_NoAs_recipe():
    from experiments.pamf_ssl_research.hard_pl_no_uas import configuration as old,validate as old_validate
    base=fixture()['2026092701']['F21']['train'];a=old(base,'/p','new');b=configuration(base,'/p','new')
    assert b['hard_pl']==a['hard_pl'] and b['hard_pl_no_uas'] and b['fork_parent_source']==a['fork_parent_source']
    assert b['hard_pl_shared_style_plan'] and b['hard_pl_mixstyle']==STYLE
    with pytest.raises(ValueError):old_validate(b)
    for change in ({'hard_pl_no_uas':False},{'hard_pl_shared_style_plan':False},{'target_inputs':'x'},
                   {'hard_pl_label_smoothing':.01},{'hard_pl_mixstyle':dict(STYLE,alpha=.2)}):
        bad=deepcopy(b);bad.update(change)
        with pytest.raises(ValueError):validate(bad)


def test_complete_matrix_reuses_same_parent_source_screen_and_queues_after_MS():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==[METHOD] and len(rt['rows'])==len(rt['source_identity_rows'])==3
    assert rt['prior_score_owner_state']==rt['MS_GPU_owner_state'] and rt['max_total_workers']==3
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert c['hard_pl_no_uas'] and c['hard_pl_shared_style_plan']
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']
        assert p['source_output']==c['output_root'] and p['source_row_id']==c['row_id']


def test_second_slot_counts_starting_MS_PID_and_requires_LS_GPU_completion():
    from experiments.pamf_ssl_research.control import pending_gpu_pids,capacity
    seed=2026092701;rt=dict(gpu_by_seed={str(seed):5},gpu_predecessor_state='l',MS_GPU_owner_state='m')
    states=dict(l=dict(jobs={'HPL_LS2-s'+str(seed):dict(status='RUNNING',pid=89)}),
                m=dict(jobs={'HPL_MS2-s'+str(seed):dict(status='RUNNING',pid=99)}))
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:states[p]):
        assert not slot_ready(rt,seed)
        states['l']['jobs']['HPL_LS2-s'+str(seed)]['status']='SCORED'
        assert slot_ready(rt,seed) and not slot_ready(rt,2026092702)
        active=[dict(gpu=5,child=SimpleNamespace(pid=101))]
        assert pending_gpu_pids(rt,(METHOD,),seed,active)=={99,101}
        with patch('experiments.pamf_ssl_research.control.subprocess.check_output',side_effect=['12\n13\n','19000']):
            assert capacity(5,pending_gpu_pids(rt,(METHOD,),seed))==(3,19000)


def test_real_checkpoint_smoke_runs_before_any_new_row():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_no_as_mixstyle_worker' in command
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_no_as_mixstyle.slot_ready',return_value=True), \
             patch('experiments.pamf_ssl_research.hard_pl_no_as_mixstyle.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,19000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            launch.assert_not_called()


def test_full_dual_report_uses_fixed_original_raw_goal():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    from tests.test_pamf_ssl_goal_counts import save
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=[METHOD]);rt['rows']=[r for r in rt['rows'] if r['method']=='PSM']
        for row in rt['rows']:
            row.update(method=METHOD,row_id=METHOD+'-s'+str(row['model_seed']))
            c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id']);save(Path(row['score_config']),c)
            path=Path(c['output_root'])/'scores.json';v=json.loads(path.read_text());v.update(run_id=RUN,row_id=row['row_id']);save(path,v)
        report(rt,root/'report');a=json.loads((root/'report/analysis.json').read_text())
        assert {v['method'] for v in a['verdicts']}=={METHOD} and not a['goal_met']
