"""Contract for the frozen source-only posterior IQ teacher cache."""
import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core

TEACHER_SOURCE = '20261106-phase1-source-posterior-teacher-s2026092701-r01'
TEACHER_COMMIT = '934112e2bfaea29f0437e8853c203c0dd25c8c37'
TEACHER_SCHEMA = 'phase1_source_posterior_positive_teacher_v1'
COVARIANCE_SOURCE = '20261106-phase1-source-covariance-posterior-s2026092701-r04'
COVARIANCE_COMMIT = '01f04f8881dfbeaaedfe8b60a966baeabe53e78d'
CONTEXT_SOURCE = '20261106-phase1-dura-deterministic-cache-s2026092701-r01'
FIT_SIZE = 8192


def validate_posterior_teacher(result, contract, fit, *, smoke=False):
    expected_result = dict(status='SOURCE_POSTERIOR_TEACHER_COMPLETE', commit=TEACHER_COMMIT,
                           exposure=3*FIT_SIZE, model_updates=0)
    expected_contract = dict(schema=TEACHER_SCHEMA, epochs=[21, 22, 23],
        covariance_source=COVARIANCE_SOURCE, covariance_commit=COVARIANCE_COMMIT,
        context_source=CONTEXT_SOURCE, cg_steps=160, positive_limit=.5,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0)
    if any(result.get(k) != v for k, v in expected_result.items()) or any(
            contract.get(k) != v for k, v in expected_contract.items()):
        raise ValueError('Fixed source-only posterior teacher provenance required')
    source_fit = np.asarray(contract.get('fit_ids', []))
    if (len(source_fit) != FIT_SIZE or len(fit) < 1 or len(fit) > FIT_SIZE or
            not np.array_equal(source_fit[:len(fit)], fit) or (not smoke and len(fit) != FIT_SIZE)):
        raise ValueError('Posterior teacher must use the matching source fit IDs')


def select_teacher_views(teacher, *, full_count, selected_count):
    if len(teacher) != 3*full_count or not 1 <= selected_count <= full_count:
        raise ValueError('Three complete ordered teacher views required')
    if selected_count == full_count:
        return teacher
    return torch.cat([teacher[i*full_count:i*full_count+selected_count] for i in range(3)])


def load_posterior_teacher(root, fit, *, smoke=False):
    folder = root/'runs'/TEACHER_SOURCE
    result = core.read(folder/'result.json')
    state = torch.load(folder/'teacher.pt', map_location='cpu', weights_only=True)
    validate_posterior_teacher(result, state['contract'], fit, smoke=smoke)
    teacher = state['teacher']
    if not isinstance(teacher, torch.Tensor) or teacher.dtype != torch.float32 or tuple(teacher.shape) != (3*FIT_SIZE, 2, 256):
        raise ValueError('Complete source posterior teacher IQ required')
    return select_teacher_views(teacher, full_count=FIT_SIZE, selected_count=len(fit))
