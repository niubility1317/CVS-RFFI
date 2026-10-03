import copy
import numpy as np
import pytest
import torch
from experiments.cvs_validdual_identity.model import build,VARIANTS,ValidHistoryFilter,context,dual_contract,RHO
from experiments.cvs_neural_residual_identity.model import build as base_build
from experiments.cvs_equivariant_identity.model import rotate_pair

@pytest.mark.parametrize('dynamic',[False,True])
def test_valid_causal_filter_matches_independent_numpy_and_bound(dynamic):
    torch.manual_seed(4);f=ValidHistoryFilter(dynamic).double();x=torch.randn(3,2,256,dtype=torch.float64)
    raw=torch.randn(3,2,33,dtype=torch.float64);c=raw/torch.linalg.vector_norm(raw,dim=1).sum(-1)[:,None,None]
    y=f.apply_kernel(x,c);z=x[:,0].numpy()+1j*x[:,1].numpy();k=c[:,0].numpy()+1j*c[:,1].numpy()
    expected=z.copy()
    for b in range(3):
        for t in range(32,256):expected[b,t]+=.5*sum(k[b,j]*z[b,t-j] for j in range(33))*float(f.taper[t-32])
    actual=y[:,0].numpy()+1j*y[:,1].numpy()
    assert np.max(abs(expected-actual))<1e-12
    assert torch.equal(y[...,:32],x[...,:32])
    assert ((y-x).flatten(1).norm(dim=1)<=RHO*x.flatten(1).norm(dim=1)+1e-10).all()

def test_context_and_filter_are_constant_phase_equivariant():
    torch.manual_seed(2);f=ValidHistoryFilter(True).double()
    with torch.no_grad():f.generator[-1].weight.normal_(std=.04)
    x=torch.randn(4,2,256,dtype=torch.float64);angle=x.new_full((4,),.73);other=rotate_pair(x,angle)
    assert torch.allclose(context(x),context(other),atol=1e-12,rtol=1e-12)
    assert torch.allclose(f(other)[0],rotate_pair(f(x)[0],angle),atol=1e-12,rtol=1e-12)

@pytest.mark.parametrize('variant',VARIANTS)
def test_own_scratch_eval_and_ce_gradients_without_cross_packet_state(variant):
    torch.set_num_threads(2);torch.manual_seed(23);m=build(variant).eval()
    torch.manual_seed(23);b=base_build('neural_residual_shallow').eval()
    x=torch.randn(4,2,256);before={k:v.clone() for k,v in m.state_dict().items()}
    assert torch.allclose(m(x),b(x),atol=2e-5,rtol=1e-5)
    with torch.no_grad():
        if m.validdual.dynamic:m.validdual.generator[-1].weight.normal_(std=.02)
        else:m.validdual.coeff_raw.normal_(std=.02)
    state={k:v.clone() for k,v in m.state_dict().items()}
    assert torch.allclose(m(x[:1]),m(x)[:1],atol=2e-5,rtol=1e-5)
    assert all(torch.equal(state[k],v) for k,v in m.state_dict().items())
    m.train();loss=torch.nn.functional.cross_entropy(m(x),torch.arange(4)%6);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    assert sum(p.numel() for p in m.parameters())==dual_contract(variant)['total_parameters']
    assert torch.isfinite(m(torch.zeros_like(x))).all()

def test_small_fir_inverse_is_representable_without_missing_history():
    # Capability only: oracle coefficients, no learned performance assertion.
    torch.manual_seed(3);x=torch.randn(2,2,256,dtype=torch.float64)
    c=.23+.09j;z=torch.complex(x[:,0],x[:,1]);received=z.clone();received[:,16:]+=c*z[:,:-16]
    iq=torch.stack((received.real,received.imag),1);f=ValidHistoryFilter(False).double()
    k=torch.zeros(2,2,33,dtype=torch.float64)
    for delay,weight in ((16,-c),(32,c*c)):
        k[:,0,delay]=weight.real/RHO;k[:,1,delay]=weight.imag/RHO
    assert torch.linalg.vector_norm(k,dim=1).sum(-1).max()<1
    out=f.apply_kernel(iq,k);corrected=torch.complex(out[:,0],out[:,1])
    # After taper and sufficient history, truncated inverse leaves c^3*x[t-48].
    assert torch.allclose(corrected[:,48:]-z[:,48:],c**3*z[:,:-48],atol=1e-12,rtol=1e-12)
    assert (corrected[:,48:]-z[:,48:]).norm()<(received[:,48:]-z[:,48:]).norm()

def test_source_config_rejects_extra_strategy_target_and_wrong_architecture():
    import json
    from pathlib import Path
    from experiments.cvs_validdual_identity.source import validate_config
    root=Path(__file__).resolve().parents[1]
    for v in VARIANTS:
        for seed in range(2026092701,2026092705):
            c=json.loads((root/'experiments/cvs_validdual_identity/configs'/f'{v}-s{seed}.json').read_text(encoding='utf-8'))
            validate_config(c)
            for key,value in [('target_truth','forbidden'),('checkpoint','old.pt'),('augmentation',True),('epochs',201),('extra_losses',['consistency']),('validdual',{})]:
                bad=copy.deepcopy(c);bad[key]=value
                with pytest.raises(ValueError):validate_config(bad)

def test_source_coefficient_telemetry_covers_all66_values():
    from experiments.cvs_validdual_identity.source import accumulate_validdual_groups,finalize_validdual_groups
    x=torch.randn(2,2,256);c=torch.randn(2,2,33)/100
    batch={'label':torch.tensor([0,0]),'receiver':torch.tensor([1,1]),'day':torch.tensor([1,1])};groups={}
    accumulate_validdual_groups(groups,batch,x,x,c,c)
    rows=finalize_validdual_groups(groups,expected_count=2,expected_groups=1)
    assert len(rows)==1 and len(rows[0]['coefficient_mean'])==66
    assert np.allclose(rows[0]['coefficient_mean'],c.flatten(1).double().mean(0).numpy())
    with pytest.raises(ValueError):finalize_validdual_groups(groups)

def test_source_dispatch_rejects_duplicate_matrix_and_target_paths(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from experiments.cvs_validdual_identity import dispatch as d
    root=Path(__file__).resolve().parents[1];monkeypatch.setattr(d,'PROJECT',str(tmp_path))
    spec=dict(run_id=d.RUN,runtime_root=str(tmp_path/'runs'/d.RUN),log_root=str(tmp_path/'logs'/d.RUN),rows=[])
    for v in VARIANTS:
        for seed in sorted(d.SEEDS):
            rid=f'{v}-s{seed}';c=json.loads((root/'experiments/cvs_validdual_identity/configs'/f'{rid}.json').read_text(encoding='utf-8'))
            c['output_root']=str(tmp_path/'runs'/d.RUN/rid/'source');p=tmp_path/f'{rid}.json';p.write_text(json.dumps(c),encoding='utf-8')
            spec['numerical_policy']=c['numerical_policy'];spec['rows'].append(dict(row_id=rid,variant=v,model_seed=seed,source_output=c['output_root'],source_config=str(p)))
    d.validate_spec(spec)
    bad=copy.deepcopy(spec);bad['rows'][-1]=bad['rows'][0]
    with pytest.raises(ValueError):d.validate_spec(bad)
    bad=copy.deepcopy(spec);bad['target_truth']='forbidden'
    with pytest.raises(ValueError):d.validate_spec(bad)
