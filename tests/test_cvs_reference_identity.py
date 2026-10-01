import copy
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from experiments.cvs_reference_identity.model import build,ExcitationResponse
from experiments.cvs_reference_identity.physics import received,TX_ROWS,RX_ROWS,highrate_reference
from experiments.cvs_rff_physics.known_excitation import TONES,SIGNS
from experiments.cvs_residual_identity.model import build as baseline

torch.set_num_threads(2)


def signal(tx=TX_ROWS[0],rx=RX_ROWS[0]):
    return torch.tensor(received(tx,rx).tolist(),dtype=torch.float64)[None]


def test_zero_residual_initialization_preserves_entire_baseline():
    torch.manual_seed(37);core=baseline('residual_fusion').eval()
    torch.manual_seed(37);model=build('reference_response').eval()
    for k,v in core.state_dict().items():assert torch.equal(v,model.core.state_dict()[k])
    x=torch.randn(3,2,256)
    with torch.no_grad():assert torch.equal(core(x),model(x))


def test_public_reference_response_shapes_and_ideal_signal():
    response=ExcitationResponse('reference_response').double()
    d=response.components(signal())
    assert d['features'].shape==(1,29)
    assert (d['relative_transfer']-1).abs().max()<2e-6
    assert d['quality'].item()==pytest.approx(1,abs=1e-10)
    assert len(list(response.parameters()))==0


def test_flat_complex_gain_phase_quotient_is_branch_only():
    response=ExcitationResponse('reference_response').double()
    x=signal(TX_ROWS[1]);z=torch.complex(x[:,0],x[:,1])*1.7*np.exp(.731j)
    y=torch.stack([z.real,z.imag],dim=1)
    assert (response(x)-response(y)).abs().max()<1e-10
    assert not response.contract()['whole_model_invariant']


def test_relative_cfo_removed_from_shape_but_explicitly_retained():
    response=ExcitationResponse('reference_response').double()
    a=response.components(signal(TX_ROWS[2],RX_ROWS[0]))
    b=response.components(signal(TX_ROWS[2],RX_ROWS[2]))
    assert (a['relative_transfer']-b['relative_transfer']).abs().max()<1e-10
    assert b['cfo'].item()*625000==pytest.approx(80000,abs=1e-7)


def test_cubic_tx_operator_has_analytic_inband_relative_response():
    # Closed-form projected cubic, including the declared matched-gain epsilon.
    response=ExcitationResponse('reference_response').double()
    kappa=-.005+.002j;tx=dict(TX_ROWS[0],cubic=kappa)
    s=highrate_reference();c=s*abs(s)**2
    ss=np.fft.fft(s)[TONES//4%80];cc=np.fft.fft(c)[TONES//4%80]
    ratio=1+kappa*cc/ss;mu=ratio.mean()
    power=np.mean(abs(ratio)**2);gain_square=abs(mu)**2/power
    expected=ratio/mu*gain_square/(gain_square+1e-6)
    actual=response.components(signal(tx))
    assert actual['selected_delay_samples'].item()==0
    assert np.max(abs(np.array(actual['relative_transfer'][0].tolist())-expected))<1e-10


def test_rx_and_channel_confounds_are_kept_as_negative_results():
    response=ExcitationResponse('reference_response').double()
    assert torch.equal(signal(TX_ROWS[3],RX_ROWS[0]),signal(TX_ROWS[0],RX_ROWS[4]))
    assert torch.allclose(signal(TX_ROWS[1],RX_ROWS[0]),signal(TX_ROWS[0],RX_ROWS[5]),atol=1e-12,rtol=1e-12)
    a=response.components(signal())['relative_transfer']
    b=response.components(signal(TX_ROWS[0],RX_ROWS[3]))['relative_transfer']
    assert (a-b).abs().mean()>.01


def test_single_packet_forward_separability_and_no_state_update():
    response=ExcitationResponse('reference_response').double().eval()
    x=torch.cat([signal(t) for t in TX_ROWS],dim=0)
    before={k:v.clone() for k,v in response.named_buffers()}
    assert torch.allclose(response(x),torch.cat([response(row[None]) for row in x]),atol=1e-12,rtol=1e-12)
    assert all(torch.equal(v,dict(response.named_buffers())[k]) for k,v in before.items())


def test_ce_activates_response_projection_after_zero_gate_first_step():
    torch.manual_seed(17);model=build('reference_response').train()
    x=torch.randn(4,2,256);y=torch.tensor([0,1,2,3]);opt=torch.optim.AdamW(model.parameters(),lr=.0002)
    for _ in range(2):
        opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x),y);loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    assert model.response_gain.detach().abs().sum()>0
    assert sum(float(p.grad.abs().sum()) for p in model.response_projection.parameters())>0


def test_frozen_cascade_diagnostics_never_train_or_assign_source_identity():
    from experiments.cvs_reference_identity.model import frozen_synthetic_diagnostics
    model=build('reference_response').eval();before=copy.deepcopy(model.state_dict())
    d=frozen_synthetic_diagnostics(model)
    assert len(d['records'])==30 and len(d['isolated_tx_changes'])==4
    assert not d['training_augmentation'] and not d['model_updated'] and not d['target_access']
    assert d['confounds']['iq_tx_rx_waveform_max_error']==0
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


def test_zero_forward_and_shape_guard():
    response=ExcitationResponse('reference_response')
    x=torch.zeros(2,2,256,requires_grad=True)
    y=response(x);assert torch.isfinite(y).all()
    y.sum().backward();assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):response(torch.zeros(1,2,128))
