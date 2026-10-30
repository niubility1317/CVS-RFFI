from types import SimpleNamespace
import os,subprocess,sys
from pathlib import Path
import pytest
import torch
from experiments.pamf_dura_direct_train import coarse_hint as trial


def test_joint30_schedule_teacher_turns_off_without_extra_epochs():
    assert trial.hint_weight(1)==1 and trial.hint_weight(10)==pytest.approx(.1)
    assert all(trial.hint_weight(e)==0 for e in range(11,31))
    assert trial.learning_rate(0)==pytest.approx(.001/247)
    assert trial.learning_rate(7409)==pytest.approx(.00002)
    with pytest.raises(ValueError):trial.hint_weight(31)
    with pytest.raises(ValueError):trial.learning_rate(7410)


def test_hint_has_no_teacher_gradients_and_zero_weight_removes_branch(monkeypatch):
    calls=[];coarse=torch.full((2,2,256),2.,requires_grad=True);parameter=torch.tensor(1.,requires_grad=True)
    terms={k:parameter.square().expand(2) for k in ('iq_nmse','prior_nmse','spectral_error','action_probe_nmse','log_variance_mse')}
    def forward(*args,**kwargs):
        assert kwargs==dict(return_coarse=True);calls.append('G');return None,terms,coarse,coarse
    monkeypatch.setattr(trial,'source_objective',forward)
    clean=torch.ones_like(coarse);reference=torch.ones_like(coarse,requires_grad=True)
    loss,metrics=trial.objective(None,None,clean,None,None,None,reference,.5)
    torch.testing.assert_close(loss,torch.full((2,),1.7))
    grad=torch.autograd.grad(loss.mean(),(parameter,coarse,reference),allow_unused=True)
    assert calls==['G'] and grad[0] is not None and grad[1].abs().sum()>0 and grad[2] is None
    loss,metrics=trial.objective(None,None,clean,None,None,None,None,0.)
    torch.testing.assert_close(loss,torch.full((2,),1.2))
    assert metrics['coarse_hint_nmse'].count_nonzero()==0
    with pytest.raises(ValueError):trial.objective(None,None,clean,None,None,None,None,1.)


def test_E30_child_modes_and_source_permissions(monkeypatch):
    base=trial.base
    for key,value in vars(base).copy().items():monkeypatch.setattr(base,key,value)
    trial.configure();config=base.configuration()
    assert (config['epochs'],config['updates'],config['L_exposures'],config['U_exposures'])==(30,7410,189000,1701000)
    assert config['training_stages']==1 and config['parameters']==266951 and not config['training_F_access']
    assert not config['target_training'] and config['ancestor_checkpoints']==[]
    assert config['teacher_model']['student_copied_weights']==0 and config['teacher_cache_inputs']==630000
    args=SimpleNamespace(root='root',commit='a'*40)
    assert all(base.command(args,m,'scratch_joint','target')[3]==trial.MODULE for m in ('worker','score'))


def test_real_module_entry_reaches_argument_parser():
    root=Path(__file__).resolve().parents[1];native=root/'.codex_tmp/dura_rf_native47/experiments/adv3b02_xuc/code'
    env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(native),str(root))))
    result=subprocess.run([sys.executable,'-m',trial.MODULE,'--help'],cwd=root,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert '--commit' in result.stdout


def test_optional_coarse_output_preserves_existing_objective_and_one_context():
    from experiments.pamf_dura_efficiency.model import direct_objective
    from experiments.pamf_action_restore.operators import Action
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.scale=torch.nn.Parameter(torch.tensor(.8));self.calls=0
            self.prior=torch.nn.ModuleList([torch.nn.Conv1d(9,2,1),torch.nn.Conv1d(9,2,1)])
        def estimate_context(self,x):
            self.calls+=1
            kernel=torch.zeros(len(x),64,dtype=torch.complex64);kernel[:,0]=1
            action=Action(kernel,torch.ones(len(x),256,dtype=torch.complex64),torch.zeros(len(x),256,dtype=torch.complex64))
            return x*self.scale,action,torch.full((len(x),),.01)
    torch.manual_seed(15);model=Tiny();x=torch.randn(2,2,256);clean=torch.randn_like(x)
    _,action,var=model.estimate_context(x);model.calls=0
    probes=[torch.randn(2,256,dtype=torch.complex64) for _ in range(2)]
    old=direct_objective(model,x,clean,action,var,probes)
    new=direct_objective(model,x,clean,action,var,probes,return_coarse=True)
    assert model.calls==2 and len(old)==2 and len(new)==4
    torch.testing.assert_close(old[0],new[0],atol=0,rtol=0)
    torch.testing.assert_close(new[3],x*model.scale,atol=0,rtol=0)
