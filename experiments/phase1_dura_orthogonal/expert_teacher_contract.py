"""Source-only frozen mixture provenance for waveform distillation."""
import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_orthogonal.dura_experts import DuraExperts, EXPERT_COUNT

EXPERT_SOURCE = '20261106-phase1-dura-expert-mixture-source-s2026092701-r01'
EXPERT_COMMIT = '6d1b1c1a5e396c4c70e5da836e116ee0bc3c431c'
FIT_SIZE = 32768
TEACHER_SCHEMA = 'phase1_dura_expert_waveform_teacher_v1'
TEACHER_SOURCE = '20261107-phase1-source-expert-teacher-s2026092701-r01'
TEACHER_COMMIT = '2737950a93abe5e0949af55e62bcc9b34a15d2c5'


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


def validate_teacher_cache(result, contract, fit, context_path, *, smoke=False):
    expected_result = dict(status='SOURCE_EXPERT_TEACHER_COMPLETE', commit=TEACHER_COMMIT,
                           exposure=3*FIT_SIZE, fit_ids=FIT_SIZE, model_updates=0,
                           TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0)
    expected_contract = dict(schema=TEACHER_SCHEMA, source=EXPERT_SOURCE, source_commit=EXPERT_COMMIT,
        epochs=[21, 22, 23], context_cache=str(context_path), deterministic_algorithms=True,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0)
    if any(result.get(k) != v for k, v in expected_result.items()) or any(
            contract.get(k) != v for k, v in expected_contract.items()):
        raise ValueError('Fixed source-only expert teacher cache required')
    source_fit = np.asarray(contract.get('fit_ids', []))
    if (len(source_fit) != FIT_SIZE or len(fit) < 1 or len(fit) > FIT_SIZE or
            not np.array_equal(source_fit[:len(fit)], fit) or (not smoke and len(fit) != FIT_SIZE)):
        raise ValueError('Teacher cache must match ordered source fit IDs')


def select_teacher_views(teacher, full_count, selected_count):
    if len(teacher) != 3*full_count or not 1 <= selected_count <= full_count:
        raise ValueError('Three complete ordered teacher views required')
    if selected_count == full_count:
        return teacher
    return torch.cat([teacher[i*full_count:i*full_count+selected_count] for i in range(3)])


def load_teacher_cache(root, fit, context_path, *, smoke=False):
    folder = root/'runs'/TEACHER_SOURCE
    result = core.read(folder/'result.json')
    payload = torch.load(folder/'teacher.pt', map_location='cpu', weights_only=True)
    validate_teacher_cache(result, payload['contract'], fit, context_path, smoke=smoke)
    teacher = payload['teacher']
    if not isinstance(teacher, torch.Tensor) or teacher.dtype != torch.float32 or tuple(teacher.shape) != (3*FIT_SIZE, 2, 256):
        raise ValueError('Complete source expert teacher IQ required')
    return select_teacher_views(teacher, FIT_SIZE, len(fit))
