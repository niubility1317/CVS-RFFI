from pathlib import Path
from types import SimpleNamespace
import os,subprocess,sys
import pytest
import torch
from experiments.pamf_dura_direct_train import prior_aux_free as trial


def test_only_final_spectral_action_terms_send_gradients(monkeypatch):
    q=torch.tensor(1.,requires_grad=True);prior=torch.tensor(2.,requires_grad=True);variance=torch.tensor(3.,requires_grad=True)
    terms=dict(iq_nmse=q.square(),spectral_error=2*q.square(),action_probe_nmse=3*q.square(),
        prior_nmse=prior.square(),log_variance_mse=variance.square())
    calls=[]
    def forward(*args,**kwargs):calls.append(1);return torch.tensor(99.),terms
    monkeypatch.setattr(trial,'source_objective',forward)
    loss,logged=trial.objective()
    torch.testing.assert_close(loss,torch.tensor(1.25))
    gradients=torch.autograd.grad(loss,(q,prior,variance),allow_unused=True)
    assert calls==[1] and logged is terms and gradients[0]>0 and gradients[1:] == (None,None)


def test_E30_routes_and_existing_physics_are_preserved(monkeypatch):
    base=trial.base
    for key,value in vars(base).copy().items():monkeypatch.setattr(base,key,value)
    trial.configure();c=base.configuration()
    assert (c['epochs'],c['updates'],c['L_exposures'],c['U_exposures'])==(30,7410,189000,1701000)
    assert c['parameters']==266951 and c['prior_networks_retained']==2 and c['CG_updates_retained']==4
    assert not c['training_F_access'] and c['teacher_model'] is None and c['ancestor_checkpoints']==[]
    assert c['allowed_GPUs']==[6] and c['max_total_visible_GPU_processes']==3
    args=SimpleNamespace(root='root',commit='a'*40)
    assert all(base.command(args,m,'scratch_joint','target')[3]==trial.MODULE for m in ('worker','score'))
    assert trial.learning_rate(7409)==pytest.approx(.00002)
    with pytest.raises(ValueError):trial.learning_rate(7410)


def test_wait_matches_only_exact_owned_predecessor_handles(tmp_path):
    root=tmp_path;cwd=root/'releases'/trial.PREDECESSOR_RELEASE
    tokens=['python','-u','-m',trial.PREDECESSOR_MODULE,'worker','--root',str(root),'--commit',trial.PREDECESSOR_COMMIT]
    assert trial.is_predecessor(tokens,cwd,root)
    for index,value in [(3,'unrelated.module'),(4,'run'),(8,'othercommit')]:
        changed=tokens.copy();changed[index]=value
        assert not trial.is_predecessor(changed,cwd,root)
    assert not trial.is_predecessor(tokens,tmp_path/'other',root)


def test_real_module_entry_reaches_parser():
    root=Path(__file__).resolve().parents[1];native=root/'.codex_tmp/dura_rf_native47/experiments/adv3b02_xuc/code'
    env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(native),str(root))))
    result=subprocess.run([sys.executable,'-m',trial.MODULE,'--help'],cwd=root,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert '--commit' in result.stdout
