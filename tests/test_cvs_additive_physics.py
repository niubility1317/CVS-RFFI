import math
import pytest
import torch
from experiments.cvs_additive_identity.model import build,VARIANTS
from experiments.cvs_additive_identity.physics import frozen_synthetic_diagnostics


@pytest.mark.parametrize('variant',VARIANTS)
def test_retained_coordinate_response_is_not_reported_as_whole_affine_invariance(variant):
    torch.manual_seed(71);model=build(variant).eval()
    with torch.no_grad():model.conditioner.weight[:,1].copy_(torch.linspace(-1,1,160))
    data=frozen_synthetic_diagnostics(model)
    assert len(data['records'])==30 and len(data['isolated_tx_changes'])==4
    assert data['phase_tolerance_pass'] and data['whole_affine_phase_invariant_claim'] is False
    assert data['coordinate_reconstruction_max_error']<1e-6
    assert 0<=data['conditioner']['delta_norm_ratio_min']<=data['conditioner']['delta_norm_ratio_max']<=.25*math.sqrt(160)+1e-6
    assert max(a['whole_logit_max_abs_error'] for a in data['phase_audit'] if a['received_cfo_hz']!=0)>1e-4
    assert data['confounds']['iq_tx_rx_waveform_max_error']<=1e-12
    assert data['confounds']['cubic_tx_rx_waveform_max_error']<=1e-12
    assert not any(data[k] for k in ('training_augmentation','target_access','model_updated','hardware_parameter_recovery'))


def test_actual_remote_source_and_inspection_scripts_compile():
    from experiments.cvs_additive_identity.publish import REMOTE,INSPECT,PROJECT
    compile(REMOTE.replace('CONFIG',repr({'project':'/p','release':'r','run':'run'})),'remote','exec')
    compile(INSPECT.replace('PROJECT',repr(PROJECT)).replace('RELEASE',repr('r')).replace('RUN',repr('run')),'inspect','exec')
