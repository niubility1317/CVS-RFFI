from types import SimpleNamespace
import torch

from experiments.pamf_dura_direct_train import rx_canonical as trial


def test_reference_is_fixed_source_shrinkage_and_only_IQ_prediction_has_gradients(monkeypatch):
    monkeypatch.setattr(trial,'_mean',torch.zeros(160));basis=torch.zeros(160,4);basis[:4]=torch.eye(4)
    monkeypatch.setattr(trial,'_basis',basis);monkeypatch.setattr(trial,'_device_constants',{})
    reference=torch.randn(2,160,requires_grad=True)
    changed=trial.canonical_reference(reference)
    original=torch.nn.functional.normalize(reference.detach(),dim=1,eps=1e-4)
    torch.testing.assert_close(changed[:,:4],.75*original[:,:4]);torch.testing.assert_close(changed[:,4:],original[:,4:])
    assert not changed.requires_grad


def test_objective_uses_one_G_and_one_F_feature_forward(monkeypatch):
    signal=torch.tensor([2.],requires_grad=True);noise=torch.tensor([3.],requires_grad=True);calls=[]
    restored=signal[:,None].expand(1,160)
    terms={k:signal.square() for k in ('iq_nmse','prior_nmse','spectral_error','action_probe_nmse')}
    terms['log_variance_mse']=noise.square()
    def source(*args,**kwargs):
        assert kwargs==dict(return_restored=True);calls.append('G');return torch.zeros(1),terms,restored
    class Reader:
        def features(self,x):calls.append('F');return x
    monkeypatch.setattr(trial.f_guided,'source_objective',source)
    monkeypatch.setattr(trial,'canonical_reference',lambda reference:torch.ones_like(reference))
    loss,logged,value=trial.objective(None,Reader(),None,torch.ones(1,2,256),None,None,None,torch.ones(1,160))
    gi,gn=torch.autograd.grad(loss,(signal,noise),allow_unused=True)
    assert calls==['G','F'] and gn is None and value is restored and gi is not None
    assert trial.METRIC in logged


def test_scratch_budget_child_modes_and_source_F_permissions_are_explicit():
    base=trial.f_guided.base
    previous={k:getattr(base,k) for k in ('RUN','configuration','command','learning_rate','direct_objective','save_model','load_model','smoke','train')}
    objective=trial.f_guided.objective;metric=trial.f_guided.GUIDANCE_METRIC
    try:
        trial.configure();c=base.configuration()
        assert c['epochs']==20 and c['updates']==4940 and c['training_stages']==1 and c['ancestor_checkpoints']==[]
        assert c['training_F_access'] and not c['hard_pseudo_TX_labels'] and c['F_feature_weight']==trial.WEIGHT
        assert c['allowed_GPUs']==[4,6,7] and c['max_total_visible_GPU_processes']==3
        assert c['F_RX_reference']['shrinkage']==.25 and c['additional_inference_branches']==0
        args=SimpleNamespace(root='root',commit='a'*40)
        assert all(base.command(args,mode,'scratch_joint','target')[3]==trial.MODULE for mode in ('worker','score'))
    finally:
        for k,v in previous.items():setattr(base,k,v)
        trial.f_guided.objective=objective;trial.f_guided.GUIDANCE_METRIC=metric
