import pytest
import torch

from experiments.phase1_dura_orthogonal.conditional_phase_oracle_probe import rotate
from experiments.phase1_dura_orthogonal.phase_supervised_dura import PhaseSupervisedDura, phase_supervision_loss


def test_phase_supervision_matches_known_effective_rotation():
    torch.manual_seed(9)
    anchor=torch.randn(4,2,256)
    variance=torch.full((4,),.5)
    clean=rotate(anchor,torch.full((4,),.3))
    baseline=phase_supervision_loss(anchor,anchor,variance,clean)
    ideal=phase_supervision_loss(rotate(anchor,torch.full((4,),.3)),anchor,variance,clean)
    assert baseline > .08 and ideal < 1e-10


def test_phase_supervision_has_gradient_and_ignores_zero_noise_gate():
    torch.manual_seed(10)
    anchor=torch.randn(4,2,256)
    clean=rotate(anchor,torch.full((4,),.2))
    model=PhaseSupervisedDura(width=4)
    output=model(anchor,anchor,anchor,torch.full((4,),.5))
    loss=phase_supervision_loss(output,anchor,torch.full((4,),.5),clean)
    loss.backward()
    assert torch.isfinite(loss) and model.summary[-1].weight.grad.abs().sum()>0
    quiet=phase_supervision_loss(output,anchor,torch.full((4,),1e-12),clean)
    assert quiet.item()==0.0


def test_supervision_uses_combined_output_phase_not_a_branch_angle():
    torch.manual_seed(11)
    anchor=torch.randn(4,2,256)
    clean=rotate(anchor,torch.full((4,),.4))
    local_rotated=rotate(anchor,torch.full((4,),.8))
    combined=.5*anchor+.5*local_rotated
    assert phase_supervision_loss(combined,anchor,torch.full((4,),.5),clean)<1e-10


def test_ground_rejects_missing_supervision_contract_and_target_reads(tmp_path, monkeypatch):
    import json
    import numpy as np
    from experiments.phase1_dura_orthogonal import ground

    folder=tmp_path/'runs'/'source'
    folder.mkdir(parents=True)
    common=dict(commit='test',epochs=5,architecture='phase_supervised_dura',objective='wide_supervised_phase',
        source_fit_size=32768,DURA_training_reads=1,DURA_frozen=True,TX_labels_read=0,F_weights_read=0,
        F_outputs_read=0,V_reads=0,target_reads=0,conditional_phase_inputs='received_anchor_innovation',
        conditional_phase_max_radians='pi/2',conditional_phase_magnitude_range=[0.9,1.1],
        conditional_phase_mix='learned_per_record',conditional_phase_noise_gate=True,
        conditional_phase_moments=[2,4],conditional_phase_moment_normalization='per_signal_power',
        source_phase_supervision_weight=1.,source_phase_supervision='gated_wrapped_full_output_angle_to_clean_oracle',
        source_phase_supervision_losses=[.1]*5,phase_target_inference_reads=0)
    summary=dict(common,status='SOURCE_TRAINED',fit_ids=32768)
    state=dict(common,schema=ground.SCHEMA,fit_ids=np.arange(32768),width=4,
               model=PhaseSupervisedDura(width=4).state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary),encoding='utf8')
    torch.save(state,folder/'final.pt')
    monkeypatch.setattr(ground,'frozen_dura',lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core,'teacher',lambda *a: (torch.nn.Identity(),None))
    assert isinstance(ground.load_fixed(tmp_path,'source','test')[0],PhaseSupervisedDura)
    summary['phase_target_inference_reads']=1
    (folder/'source_summary.json').write_text(json.dumps(summary),encoding='utf8')
    with pytest.raises(ValueError,match='supervised phase'):
        ground.load_fixed(tmp_path,'source','test')
