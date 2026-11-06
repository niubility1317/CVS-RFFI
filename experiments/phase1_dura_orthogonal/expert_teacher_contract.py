"""Source-only frozen mixture provenance for waveform distillation."""
import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_orthogonal.dura_experts import DuraExperts, EXPERT_COUNT

EXPERT_SOURCE = '20261106-phase1-dura-expert-mixture-source-s2026092701-r01'
EXPERT_COMMIT = '6d1b1c1a5e396c4c70e5da836e116ee0bc3c431c'
FIT_SIZE = 32768
TEACHER_SCHEMA = 'phase1_dura_expert_waveform_teacher_v1'


def validate_expert_checkpoint(summary, state, fit):
    expected = dict(commit=EXPERT_COMMIT, architecture='dura_experts', objective='wide_mixture_guard',
        epochs=5, source_fit_size=FIT_SIZE, source_expert_count=EXPERT_COUNT, deterministic_algorithms=True,
        DURA_training_reads=1, DURA_frozen=True, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0)
    if (summary.get('status') != 'SOURCE_TRAINED' or summary.get('fit_ids') != FIT_SIZE or
            state.get('schema') != 'phase1_dura_orthogonal_source_v1' or any(
                obj.get(k) != v for obj in (summary, state) for k, v in expected.items())):
        raise ValueError('Source-only fixed expert checkpoint required')
    source_fit = np.asarray(state.get('fit_ids', []))
    if (len(source_fit) != FIT_SIZE or len(fit) < 1 or len(fit) > FIT_SIZE or
            not np.array_equal(source_fit[:len(fit)], fit)):
        raise ValueError('Matching ordered source fit IDs required')


def load_expert_teacher(root, fit, device):
    folder = root/'runs'/EXPERT_SOURCE
    summary = core.read(folder/'source_summary.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    validate_expert_checkpoint(summary, state, fit)
    model = DuraExperts(width=state['width'], taps=9)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)
