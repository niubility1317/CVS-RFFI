from pathlib import Path
from types import SimpleNamespace
import torch
from experiments.pamf_dura_direct_train import signature_guided as m


def test_sig_guide_is_added_once_and_extra_F_path_preserves_probe_rng(monkeypatch):
    restored=torch.ones(2,2,256,requires_grad=True)
    physical={k:torch.full((2,),v) for k,v in dict(iq_nmse=1.,prior_nmse=2.,spectral_error=3.,action_probe_nmse=4.,log_variance_mse=1e6).items()}
    monkeypatch.setattr(m,'source_objective',lambda *a,**kw:(None,physical,restored))
    def guide(reader,r,*args):
        torch.randn(13)
        return .001*r.sum((1,2)),dict(feature_error=torch.ones(2),weight=torch.ones(2),raw_pressure=torch.ones(2),bounded_pressure=torch.full((2,),.5),pressure_defined=torch.ones(2,dtype=torch.bool))
    monkeypatch.setattr(m,'signature_guide',guide)
    before=torch.get_rng_state().clone();loss,terms,_=m.objective(None,None,restored,restored.detach(),None,None,None,None)
    assert torch.equal(before,torch.get_rng_state())
    torch.testing.assert_close(loss,torch.full((2,),1.+.1*2+.05*3+.05*4+.512))
    loss.mean().backward();torch.testing.assert_close(restored.grad,torch.full_like(restored,.0005))


def test_final_G_loader_defaults_to_E30_but_allows_real_smoke_E0(monkeypatch):
    seen=[]
    monkeypatch.setattr(m.f_guided,'load_model',lambda *a,**kw:seen.append(kw['epoch']))
    m.load_model('path','cpu',arm='scratch_joint',commit='commit')
    m.load_model('path','cpu',arm='scratch_joint',commit='commit',epoch=0)
    assert seen==[30,0]


def test_single_E30_source_F_permissions_and_automatic_score_routes():
    m.configure();cfg=m.base.configuration()
    assert (cfg['epochs'],cfg['updates'],cfg['L_exposures'],cfg['U_exposures'])==(30,7410,189000,1701000)
    assert cfg['training_stages']==1 and cfg['training_F_access'] and cfg['F_guided_training']
    assert cfg['F_prefix_training_inputs']==cfg['F_input_VJP_inputs']==1890000 and cfg['F_full_training_inputs']==0
    assert cfg['F_clean_cache_shape']==[63000,160] and cfg['F_guidance_IQ_gradient_cap']==.5
    assert not cfg['source_qualification_claim'] and cfg['diagnostic_primary_FAILs_preserved']
    args=SimpleNamespace(root=Path('/project'),commit='commit')
    for mode,role in [('worker',None),('score','source'),('score','target')]:
        command=m.base.command(args,mode,'scratch_joint',role)
        assert m.MODULE in command and mode in command
    assert cfg['all_versions_automatic_formal'] and cfg['target_count']==168000 and len(cfg['views'])==7
