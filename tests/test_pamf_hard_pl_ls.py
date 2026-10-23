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
from experiments.pamf_ssl_research.hard_pl_ls import RUN,configuration,validate,slot_ready
from experiments.pamf_ssl_research.hard_pl_ls_publish import prepare
from experiments.pamf_ssl_research.hard_pl_objective import HardPLObjective
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def test_only_U_label_smoothing_changes_from_ordinary_weight2():
    from experiments.pamf_ssl_research.hard_pl import configuration as original,validate as old_validate
    base=fixture()['2026092701']['F21']['train'];a=original(base,'HPL2','/p','new');b=configuration(base,'/p','new')
    assert b['hard_pl']==a['hard_pl'] and b['fork_parent_source']==a['fork_parent_source']
    assert b['hard_pl_label_smoothing']==.01 and b['hard_pl']['threshold']==.95 and b['hard_pl']['weight']==2
    with pytest.raises(ValueError):old_validate(b)
    for change in ({'hard_pl_label_smoothing':0.},{'hard_pl_no_uas':True}):
        c=deepcopy(b);c.update(change)
        with pytest.raises(ValueError):validate(c)


def test_exact_smoothed_masked_CE_and_rejected_sample_gradients():
    class Student(nn.Module):
        def __init__(self):
            super().__init__();self.logits=nn.Parameter(torch.arange(24,dtype=torch.float32).reshape(4,6)/5)
            self.teacher=torch.tensor([[10.,0,0,0,0,0],[0.,0,0,0,0,0],[0.,10,0,0,0,0],[0.,0,0,0,0,0]])
            self._cd_U_seen=self._cd_U_accepted=0
        def forward(self,x):return self.teacher
        def forward_identity_only(self,x,domain_labels=None):return dict(tx_logits=self.logits)
    model=Student().train();batch=(torch.zeros(4,2,8),torch.full((4,),-1),torch.zeros(4,dtype=torch.long),{})
    model._cd_U_loader=[batch];model._cd_U_iter=iter([batch])
    objective=HardPLObjective(dict(hard_pl=dict(method='HPL2',threshold=.95,weight=2.),hard_pl_label_smoothing=.01))
    loss=objective.loss(model,'cpu');mask=torch.tensor([1.,0,1,0]);labels=model.teacher.argmax(1)
    expected=2*(F.cross_entropy(model.logits,labels,reduction='none',label_smoothing=.01)*mask).mean()
    torch.testing.assert_close(loss,expected)
    gradients=torch.autograd.grad(loss,model.logits)[0]
    assert torch.count_nonzero(gradients[[1,3]])==0 and torch.count_nonzero(gradients[[0,2]])>0
    assert objective.metrics['HPL_label_smoothing']==.01 and objective.state_dict()['label_smoothing']==.01
    assert objective.accepted==2 and objective.student_views==4 and objective.physical_seen==objective.raw_As_seen==0


def test_three_seed_both_readouts_chain_and_no_hidden_U_truth():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==['HPL_LS2'] and len(rt['rows'])==3 and rt['max_total_workers']==3
    assert rt['prior_score_owner_state']==rt['gpu_predecessor_state'] and 'noUAs' in rt['gpu_predecessor_state']
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert c['hard_pl_label_smoothing']==.01 and c['hard_pl']['method']=='HPL2'
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']
        assert c['output_root']==p['source_output'] and c['row_id']==p['source_row_id']


def test_predecessor_GPU_handoff_and_pre_CUDA_capacity_reservation():
    from experiments.pamf_ssl_research.control import pending_gpu_pids,capacity
    seed=2026092701;rt=dict(gpu_by_seed={str(seed):5},gpu_predecessor_state='n',hard_PL_GPU_owner_state='h')
    n=dict(jobs={'HPL_NO_UAS-s'+str(seed):dict(status='RUNNING',pid=98)})
    h=dict(jobs={'HPLT1U-s'+str(seed):dict(status='RUNNING',pid=99)})
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:n if p=='n' else h):
        assert not slot_ready(rt,seed)
        n['jobs']['HPL_NO_UAS-s'+str(seed)]['status']='PREDICTIONS_COMPLETE'
        assert slot_ready(rt,seed) and not slot_ready(rt,2026092702)
        own=[dict(gpu=5,child=SimpleNamespace(pid=101))]
        assert pending_gpu_pids(rt,('HPL_LS2',),seed,own)=={99,101}
        with patch('experiments.pamf_ssl_research.control.subprocess.check_output',side_effect=['12\n13\n','19000']):
            assert capacity(5,pending_gpu_pids(rt,('HPL_LS2',),seed))==(3,19000)
        h['jobs']['HPLT1U-s'+str(seed)]['status']='SCORED'
        assert pending_gpu_pids(rt,('HPL_LS2',),seed,own)=={101}


def test_real_smoke_is_first_available_slot_step_and_dedicated_worker():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_ls_worker' in command
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_ls.slot_ready',side_effect=[False,True]), \
             patch('experiments.pamf_ssl_research.hard_pl_ls.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,19000)), \
             patch('experiments.pamf_ssl_research.control.time.sleep') as sleep, \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            sleep.assert_called_once_with(20);launch.assert_not_called()


def test_full_report_keeps_original_raw_goal_and_smoothed_method_identity():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=['HPL_LS2'])
        rt['rows']=[r for r in rt['rows'] if r['method']=='PSM']
        for row in rt['rows']:
            row.update(method='HPL_LS2',row_id='HPL_LS2-s'+str(row['model_seed']))
            c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id']);save(Path(row['score_config']),c)
            path=Path(c['output_root'])/'scores.json';v=json.loads(path.read_text());v.update(run_id=RUN,row_id=row['row_id']);save(path,v)
        report(rt,root/'report');analysis=json.loads((root/'report/analysis.json').read_text())
        assert {v['method'] for v in analysis['verdicts']}=={'HPL_LS2'} and not analysis['goal_met']
