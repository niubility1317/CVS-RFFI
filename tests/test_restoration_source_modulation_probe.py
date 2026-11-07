import torch

from experiments.pamf_phase3_restoration.covariance_inverse import dense_action
from experiments.pamf_phase3_restoration.factorized_action_loss import aligned_factors
from experiments.pamf_phase3_restoration.source_modulation_probe import modulation_controls


def test_modulation_controls_isolate_phase_gain_main_and_image():
    dtype = torch.complex128
    h = torch.tensor([[1+.2j,.4-.1j]],dtype=dtype)
    main = torch.tensor([[.8+.2j,1+.4j,1.1+.1j,.9-.2j]],dtype=dtype)
    image = torch.tensor([[.1+.05j,.05+.02j,.03+.04j,.04+.07j]],dtype=dtype)
    gain = torch.tensor(1.3+.4j,dtype=dtype)
    residual = torch.tensor([[.04-.03j,-.01+.01j,.02+.03j,-.01+.01j]],dtype=dtype)
    residual = residual-main*(main.conj()*residual).sum()/main.abs().square().sum()
    true_main = gain*main+residual
    true_image = image * (2+.1j)

    actions,fit = modulation_controls((h,main,image),(h,true_main,true_image))
    torch.testing.assert_close(fit, gain[None])
    torch.testing.assert_close(dense_action(*actions['main_phase']),dense_action(h,main*gain/gain.abs(),image))
    torch.testing.assert_close(dense_action(*actions['main_complex_gain']),dense_action(h,main*gain,image))
    torch.testing.assert_close(dense_action(*actions['true_main']),dense_action(h,true_main,image))
    torch.testing.assert_close(dense_action(*actions['true_image']),dense_action(h,main,true_image))


def test_modulation_controls_keep_same_actions_after_equivalent_factor_gauge():
    h = torch.tensor([[.8+.1j,.3-.2j]],dtype=torch.complex128)
    main = torch.tensor([[.9-.1j,.7+.3j,1.4-.2j]],dtype=h.dtype)
    image = torch.tensor([[.03+.02j,.01+.03j,.02-.04j]],dtype=h.dtype)
    estimated = h, main, image
    target = h.flip(1), main.roll(1,dims=1), image.flip(1)
    def regauge(factors, gain):
        h,m,i = factors
        return h*gain, m/gain, i/gain.conj()
    a,b,_ = aligned_factors(estimated,target)
    control,gain = modulation_controls(a,b)
    ra,rb,_ = aligned_factors(regauge(estimated,torch.tensor(1.2+.6j,dtype=h.dtype)),
                              regauge(target,torch.tensor(-.5+.8j,dtype=h.dtype)))
    other,other_gain = modulation_controls(ra,rb)
    torch.testing.assert_close(gain,other_gain)
    for name in control:
        torch.testing.assert_close(dense_action(*control[name]),dense_action(*other[name]))


def test_reject_non_frozen_parent_before_loading_control_predictions(tmp_path):
    import json
    import pytest
    from experiments.pamf_phase3_restoration.source_modulation_probe import CONTROL_COMMIT, load_controls

    result = dict(status='SOURCE_ACTION_FACTOR_COMPLETE',code_commit=CONTROL_COMMIT,source_V_records=480,
        TX_labels_read=0,target_inputs=1)
    (tmp_path/'results.json').write_text(json.dumps(result),encoding='utf-8')
    (tmp_path/'complete.json').write_text('{"status":"ARTIFACTS_COMPLETE"}',encoding='utf-8')
    with pytest.raises(ValueError,match='Exact frozen source factor controls'):
        load_controls(tmp_path,None,None)
