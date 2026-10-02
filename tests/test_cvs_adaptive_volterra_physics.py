import pytest
import torch
from experiments.cvs_adaptive_volterra_identity.model import build,VARIANTS
from experiments.cvs_adaptive_volterra_identity.physics import frozen_synthetic_diagnostics

@pytest.mark.parametrize('variant',VARIANTS)
def test_frozen_cascade_measures_waveform_covariance_not_logit_invariance(variant):
    torch.manual_seed(71);model=build(variant).eval()
    state={k:v.clone() for k,v in model.state_dict().items()}
    d=frozen_synthetic_diagnostics(model)
    assert len(d['records'])==30 and len(d['isolated_tx_changes'])==4
    assert d['phase_tolerance_pass'] and not d['whole_affine_phase_invariant_claim']
    assert d['alignment_strength']==float(model.alignment_strength().detach())
    assert d['normalization']['blocks']==6 and len(d['normalization']['records'])==6
    assert all(a['eligible_packets']==30 and a['relative_energy_fraction_max_error']<1e-6 for a in d['normalization']['records'])
    assert d['coordinate_reconstruction_max_error']<1e-6 and d['sample_amplitude_max_error']<1e-6
    for a in d['phase_audit']:
        assert a['covariance_eligible_count']==30
        assert a['partial_waveform_covariance_max_error']<1e-5
    assert max(a['whole_logit_max_abs_error'] for a in d['phase_audit'] if a['received_cfo_hz']!=0)>1e-4
    assert max(d['confounds'].values())<=1e-12
    assert not any(d[k] for k in ('training_augmentation','target_access','model_updated','hardware_parameter_recovery'))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())

def test_remote_publish_collect_scripts_compile():
    from experiments.cvs_adaptive_volterra_identity.publish import REMOTE,INSPECT,PROJECT
    from experiments.cvs_adaptive_volterra_identity.collect import REMOTE as COLLECT
    compile(REMOTE.replace('CONFIG',repr({'project':'/p','release':'r','run':'run'})),'remote','exec')
    compile(INSPECT.replace('PROJECT',repr(PROJECT)).replace('RELEASE',repr('r')).replace('RUN',repr('run')),'inspect','exec')
    compile(COLLECT.replace('CONFIG',repr({'project':'/p','run':'r'})),'collect','exec')
