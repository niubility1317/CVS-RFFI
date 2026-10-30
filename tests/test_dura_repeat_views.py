from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_dura_direct_train import repeat_views as trial,enhanced
from experiments.pamf_dura_efficiency.model import scratch_model


@pytest.fixture
def restored(monkeypatch):
    for module in (trial.base,trial.queue):
        for key,value in vars(module).copy().items():monkeypatch.setattr(module,key,value)


def test_enhancement_cpu_factory_preserves_cuda_probe_seed_and_G_state(restored,monkeypatch):
    seeds=[];monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda seed:seeds.append(int(seed)))
    torch.manual_seed(2026092701);before=torch.get_rng_state();seeds.clear()
    compact=scratch_model(2026092701);assert seeds==[2026092701]
    enhanced.configure();seeds.clear();model=trial.base.scratch_model(2026092701)
    assert seeds==[2026092701] and torch.equal(before,torch.get_rng_state())
    values={k:v for k,v in model.state_dict().items() if not k.startswith('enhancer.')}
    assert set(values)==set(compact.state_dict()) and all(torch.equal(v,compact.state_dict()[k]) for k,v in values.items())
    assert sum(p.numel() for p in model.enhancer.parameters())==930


def test_pure_control_has_matching_budget_seed_and_original_source_paths(restored,tmp_path):
    trial.configure();base=trial.base;c=base.configuration();args=SimpleNamespace(root=tmp_path,commit='a'*40)
    assert (c['epochs'],c['updates'],c['corruption_epochs'],c['parameters'],c['training_stages'])==(30,7410,20,266951,1)
    assert c['CUDA_probe_seed']==2026092701 and c['numpy_shuffle_seed']==2026103004
    assert not c['training_F_access'] and c['teacher_model'] is None and c['allowed_GPUs']==[4]
    assert base.paths(tmp_path)['cache']==tmp_path/'runs'/base.ACTION/'cache'
    assert all(base.command(args,m,'scratch_joint','target')[3]==trial.MODULE for m in ('worker','score'))
    tokens=['python','-u','-m',trial.FRESH_MODULE,'score','--root',str(tmp_path),'--commit',trial.FRESH_COMMIT]
    assert trial.queue.is_predecessor(tokens,tmp_path/'releases'/trial.FRESH_RELEASE,tmp_path)
    tokens[3]='unrelated.module';assert not trial.queue.is_predecessor(tokens,tmp_path/'releases'/trial.FRESH_RELEASE,tmp_path)


def test_control_checkpoint_and_final_LR_are_strict(restored,tmp_path):
    trial.configure();base=trial.base;model=base.scratch_model(2026092701).eval();x=torch.ones(1,2,256)
    with torch.no_grad():expected=model(x)
    checkpoint=tmp_path/'model.pt';base.save_model(checkpoint,model,'scratch_joint',0,0,'a'*40)
    loaded=base.load_model(checkpoint,'cpu',arm='scratch_joint',commit='a'*40,epoch=0)
    with torch.no_grad():torch.testing.assert_close(loaded(x),expected,atol=0,rtol=0)
    assert trial.learning_rate(7409)==pytest.approx(.00002)
    with pytest.raises(ValueError):base.load_model(checkpoint,'cpu',arm='scratch_joint',commit='b'*40,epoch=0)
