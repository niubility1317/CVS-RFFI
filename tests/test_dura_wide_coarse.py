import copy
from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_dura_direct_train import wide_coarse as trial
from experiments.pamf_dura_direct_train.evaluation import conv_linear_mac


@pytest.fixture
def configured(monkeypatch):
    for module in (trial.base,trial.hint,trial.queue):
        for key,value in vars(module).copy().items():monkeypatch.setattr(module,key,value)
    trial.configure()
    return trial.base


def test_only_coarse_capacity_changes_and_rng_is_preserved():
    torch.manual_seed(23);before=torch.random.get_rng_state()
    wide=trial.scratch_model(2026092701);torch.testing.assert_close(torch.random.get_rng_state(),before)
    compact=trial.compact_scratch(2026092701)
    assert sum(p.numel() for p in wide.parameters())==1466771
    assert sum(p.numel() for p in wide.frozen.coarse.parameters())==1305958
    for part in ('estimator',):
        a=getattr(wide.frozen,part).state_dict();b=getattr(compact.frozen,part).state_dict()
        assert all(torch.equal(value,b[key]) for key,value in a.items())
    assert all(torch.equal(value,compact.prior.state_dict()[key]) for key,value in wide.prior.state_dict().items())
    assert wide.initialization_report['checkpoint_sources']==[] and wide.initialization_report['copied_teacher_parameters']==0


def test_wide_checkpoint_strict_roundtrip_and_full_epoch_metadata(configured,tmp_path):
    args=SimpleNamespace(root=tmp_path,commit='a'*40);model=trial.scratch_model(2026092701).eval()
    x=torch.randn(1,2,256,generator=torch.Generator().manual_seed(7))
    with torch.no_grad():expected=model(x)
    checkpoint=tmp_path/'model.pt';configured.save_model(checkpoint,model,'scratch_joint',0,0,args.commit)
    loaded=configured.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    with torch.no_grad():torch.testing.assert_close(loaded(x),expected,atol=0,rtol=0)
    c=configured.configuration()
    assert (c['epochs'],c['updates'],c['parameters'])==(30,7410,1466771)
    assert c['teacher_cache_inputs']==0 and c['historic_teacher_cache_inputs']==630000
    assert c['allowed_GPUs']==[6] and not c['training_F_access']
    assert all(configured.command(args,m,'scratch_joint','target')[3]==trial.MODULE for m in ('worker','score'))


def test_cache_provenance_rejects_wrong_teacher_or_hidden_data():
    c=dict(status='SOURCE_COARSE_IQ_CACHE_COMPLETE',teacher_run=trial.hint.TEACHER_RUN,teacher_commit=trial.hint.TEACHER_COMMIT,
        teacher_component='frozen.coarse',teacher_used_parameters=1305958,teacher_loaded_parameters=4665073,
        teacher_component_epochs=20,teacher_bundle_component_epochs=60,epochs=list(range(1,11)),count_per_epoch=63000,
        teacher_inputs=630000,numeric_bytes=1290240000,teacher_updates=0,teacher_unchanged=True,F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0)
    launch=dict(commit=trial.CACHE_COMMIT);plan=dict(source_contract='original7/63/30')
    done=dict(status='SOURCE_TRAINED',epoch=30,updates=7410,training_F_access=False,training_TX_labels_read=0,V_reads=0,target_reads=0,teacher_updates=0)
    trial.validate_cache_metadata(c,launch,plan,done)
    for key,value in (('teacher_commit','bad'),('F_reads',1),('TX_labels_read',1),('target_reads',1)):
        changed=copy.deepcopy(c);changed[key]=value
        with pytest.raises(ValueError):trial.validate_cache_metadata(changed,launch,plan,done)


def test_profile_macs_count_repeated_shared_module_and_remove_hooks():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.conv=torch.nn.Conv1d(2,4,3,padding=1);self.linear=torch.nn.Linear(4,2)
        def forward(self,x):
            y=self.conv(x)+self.conv(x);return self.linear(y.transpose(1,2))
    model=Toy();assert conv_linear_mac(model,torch.ones(1,2,8))==448
    assert all(not m._forward_hooks for m in model.modules())


def test_actual_graph_cost_matches_compact_reference_and_counts_expansion():
    x=torch.randn(1,2,256,generator=torch.Generator().manual_seed(3))
    compact=trial.compact_scratch(2026092701).eval();wide=trial.scratch_model(2026092701).eval()
    assert conv_linear_mac(compact,x)==25068072
    assert conv_linear_mac(wide,x)>25068072


def test_evaluation_wait_is_bound_to_actual_strong_phase_process(configured,tmp_path):
    tokens=['python','-u','-m',trial.PREDECESSOR_MODULE,'score','--root',str(tmp_path),'--commit',trial.PREDECESSOR_COMMIT]
    cwd=tmp_path/'releases'/trial.PREDECESSOR_RELEASE
    assert trial.queue.is_predecessor(tokens,cwd,tmp_path)
    tokens[3]='unrelated.module';assert not trial.queue.is_predecessor(tokens,cwd,tmp_path)
