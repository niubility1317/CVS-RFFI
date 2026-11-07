import json
import sys

import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal.conditional_iq_critic import ConditionalIQCritic, update_critic, OBJECTIVE
from experiments.phase1_dura_orthogonal import train_adversarial_source as trainer, ground, controller


def fields(count=4):
    generator=torch.Generator().manual_seed(28)
    x, anchor, direction, clean=(torch.randn(count,2,256,generator=generator) for _ in range(4))
    return x,anchor,direction,torch.full((count,),.1),clean


def test_critic_update_is_separate_and_generator_loss_has_no_critic_or_clean_gradients():
    x,anchor,_,variance,clean=fields()
    clean.requires_grad_(True)
    restored=anchor.clone().requires_grad_(True)
    critic=ConditionalIQCritic()
    optimizer=torch.optim.AdamW(critic.parameters(),lr=.0001,betas=(.5,.9))
    before=critic.head.weight.detach().clone()
    dloss,gloss=update_critic(critic,optimizer,x,anchor,restored,variance,clean)
    assert torch.isfinite(dloss) and not torch.equal(before,critic.head.weight)
    assert restored.grad is None and clean.grad is None
    buffers={key:value.clone() for key,value in critic.named_buffers()}
    gloss.backward()
    assert restored.grad is not None and restored.grad.abs().sum()>0
    assert clean.grad is None and all(p.grad is None and not p.requires_grad for p in critic.parameters())
    assert all(torch.equal(value,dict(critic.named_buffers())[key]) for key,value in buffers.items())


def test_nonfinite_critic_input_prevents_optimizer_update():
    x,anchor,_,variance,clean=fields()
    critic=ConditionalIQCritic()
    optimizer=torch.optim.AdamW(critic.parameters())
    before=[p.detach().clone() for p in critic.parameters()]
    with pytest.raises(FloatingPointError,match='nonfinite'):
        update_critic(critic,optimizer,x,anchor,anchor*float('nan'),variance,clean)
    assert all(torch.equal(a,b) for a,b in zip(before,critic.parameters()))
    assert not optimizer.state


def test_actual_trainer_smoke_and_ground_provenance(monkeypatch,tmp_path):
    data=fields(96)
    models=[]
    original=trainer.base.DuraMultiscaleSpectral
    class Capture(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            models.append(self)
    monkeypatch.setattr(trainer.base,'DuraMultiscaleSpectral',Capture)
    monkeypatch.setattr(trainer.base,'training_environments',lambda *a:None)
    monkeypatch.setattr(trainer.base,'source_inputs',lambda *a:(np.arange(32),None,None,None,None,None))
    monkeypatch.setattr(trainer.base,'load_source_training',lambda *a,**k:(None,None,None,np.arange(32)))
    monkeypatch.setattr(trainer.base,'frozen_dura',lambda *a:torch.nn.Identity().requires_grad_(False))
    monkeypatch.setattr(trainer.base,'prepare',lambda *a,**k:(data,{'synthetic':96}))
    out=tmp_path/'smoke'
    monkeypatch.setattr(sys,'argv',['trainer','--root',str(tmp_path),'--out',str(out),'--commit','test',
        '--device','cpu','--smoke','--smoke-fit','32','--deterministic','--objective',OBJECTIVE,'--architecture','dura_multiscale'])
    trainer.main()
    summary=json.loads((out/'smoke.json').read_text())
    assert summary['status']=='SOURCE_TRAINING_SMOKE_PASS'
    assert summary['TX_labels_read']==summary['F_weights_read']==summary['F_outputs_read']==summary['V_reads']==summary['target_reads']==0
    assert len(summary['source_critic_losses'])==1 and summary['critic_inference_reads']==0
    assert models[0].wave.weight.abs().sum()>0
    summary.update(status='SOURCE_TRAINED',epochs=5,fit_ids=32768)
    state=dict(summary,schema=trainer.base.SCHEMA,model=models[0].state_dict(),width=32)
    state['fit_ids']=np.arange(32768)
    monkeypatch.setattr(ground.core,'read',lambda *a:summary)
    monkeypatch.setattr(ground.torch,'load',lambda *a,**k:state)
    monkeypatch.setattr(ground,'frozen_dura',lambda *a:torch.nn.Identity())
    monkeypatch.setattr(ground.core,'teacher',lambda *a:(torch.nn.Identity(),None))
    ground.load_fixed(tmp_path,'source','test')
    state['critic_inference_reads']=1
    with pytest.raises(ValueError,match='conditional adversary'):
        ground.load_fixed(tmp_path,'source','test')


def test_controller_uses_dedicated_trainer_for_both_source_stages(monkeypatch,tmp_path):
    monkeypatch.setattr(sys,'argv',['controller','--root',str(tmp_path),'--source','source','--ground','ground',
        '--commit','test','--device','cpu','--deterministic','--objective',OBJECTIVE,'--architecture','dura_multiscale'])
    commands=[]
    def fake_run(command,**kwargs):
        commands.append(command)
        assert 'experiments.phase1_dura_orthogonal.train_adversarial_source' in command
        if len(commands)==1:
            out=tmp_path/'runs/source-training-smoke'
            out.mkdir(parents=True)
            (out/'smoke.json').write_text(json.dumps({'status':'SOURCE_TRAINING_SMOKE_PASS','F_weights_read':0}))
            return type('Result',(),{'returncode':0})()
        raise RuntimeError('test stops before actual training')
    monkeypatch.setattr(controller.subprocess,'run',fake_run)
    with pytest.raises(RuntimeError,match='test stops'):
        controller.main()
    assert len(commands)==2
