from types import SimpleNamespace
import numpy as np
import torch
from experiments.pamf_dura_direct_train import runner as base,spectral_mix
from experiments.pamf_dura_efficiency.spectral_style import bank_from_psd
from experiments.pamf_restoration.models import complex_iq,pair


def sampler_fixture():
    sessions=np.asarray([(r,d) for r in (1,3,4,6,8) for d in (1,2,3)],np.int64)
    power=torch.stack([torch.linspace(1,1+i*.1,16) for i in range(15)])
    gain,_=bank_from_psd(power)
    source=sessions[np.arange(64)%15]
    roles=np.asarray(['L_s']*20+['U_s']*44)
    return spectral_mix.SpectralPairing(np.asarray(gain.tolist(),np.float32),sessions,source[:,0],source[:,1],roles,'cpu')


def test_partial_augmentation_keeps_unselected_rows_and_same_noncommuting_noise():
    generator=torch.Generator().manual_seed(23)
    clean=torch.randn(64,2,16,generator=generator)
    noise=torch.randn(clean.shape,generator=generator)*.05
    phase=torch.exp(1j*torch.linspace(0,3,16))
    def action(z):return phase*z+.2*z.conj()
    received=pair(action(complex_iq(clean)))+noise
    sampler=sampler_fixture();before=torch.get_rng_state().clone()
    observed,reference=sampler(received,clean,action,np.arange(64))
    mask=np.random.default_rng(spectral_mix.SEED).random(64)<.5
    untouched=torch.tensor(np.flatnonzero(~mask).tolist())
    assert torch.equal(reference[untouched],clean[untouched]) and torch.equal(observed[untouched],received[untouched])
    torch.testing.assert_close(observed-pair(action(complex_iq(reference))),noise,atol=1e-6,rtol=2e-5)
    torch.testing.assert_close(reference.square().mean((1,2)),clean.square().mean((1,2)),atol=3e-7,rtol=2e-6)
    assert torch.equal(before,torch.get_rng_state())
    meta=sampler.summary();draws=np.asarray(meta['cross_session_draw_counts']);sessions=np.asarray(meta['sessions'])
    assert meta['seen']==64 and meta['selected']==int(mask.sum()) and sum(meta['selected_by_role'].values())==int(mask.sum())
    assert draws.sum()==meta['selected'] and draws[sessions[:,0,None]==sessions[None,:,0]].sum()==0


def test_smoke_activates_augmentation_and_training_sampler_restarts_exactly():
    one=sampler_fixture();two=sampler_fixture()
    x=torch.ones(2,2,16)
    def action(z):return z
    y,reference=one(x,x,action,np.arange(2))
    y2,reference2=two(x,x,action,np.arange(2))
    assert one.selected==2
    assert torch.equal(y,y2) and torch.equal(reference,reference2) and one.summary()==two.summary()


def test_recipe_routes_children_with_original_budget_and_no_teacher():
    keys=('RUN','configuration','command','learning_rate','direct_objective','training_pair','smoke','train')
    original={key:getattr(base,key) for key in keys}
    try:
        spectral_mix.configure();config=base.configuration()
        assert (config['epochs'],config['updates'],config['training_stages'])==(20,4940,1)
        assert config['source_pair_probability']==.5 and config['G_forwards_per_training_visit']==1
        assert config['ancestor_checkpoints']==[] and not config['training_F_access'] and config['learned_teachers']==0
        assert config['allowed_GPUs']==[4,6,7] and config['max_total_visible_GPU_processes']==3
        args=SimpleNamespace(root='root',commit='a'*40)
        assert all(base.command(args,mode,'scratch_joint','target')[3]==spectral_mix.MODULE for mode in ('worker','score'))
    finally:
        for key,value in original.items():setattr(base,key,value)
