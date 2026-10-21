"""Exact retained DURA ancestry: eight same-contract, fixed-budget stages."""
from copy import deepcopy
import torch

STAGES = ('coarse', 'action', 'r17', 'r38', 'r44', 'r46', 'r65', 'r69')


def construct(stage, parent=None):
    if stage not in STAGES:
        raise ValueError('Registered DURA lineage stage required')
    if stage == 'coarse':
        from experiments.pamf_reception_restore.models import ComplexTFUNet
        return ComplexTFUNet()
    if parent is None:
        raise ValueError('A same-contract immediate parent is required')
    parent = parent.cpu().eval().requires_grad_(False)
    if stage == 'action':
        from experiments.pamf_action_restore.models import ActionRestorer
        return ActionRestorer(deepcopy(parent))
    if stage == 'r17':
        from experiments.pamf_unrolled_restore.models import UnrolledRestorer
        return UnrolledRestorer('unroll_unshared', deepcopy(parent))
    if stage == 'r38':
        from experiments.pamf_iq_joint_compact.models import compact_from_teacher
        return compact_from_teacher(parent, 'dense_half')
    if stage == 'r44':
        from experiments.pamf_r38_compression_methods.models import compact_from_parent
        return compact_from_parent(parent, 'group_pruned')
    if stage == 'r46':
        from experiments.pamf_r44_pruned_refine.models import compact_from_parent
        return compact_from_parent(parent, 'prune_refine')
    if stage == 'r65':
        from experiments.pamf_r46_delayed_prune.models import dense_from_parent
        return dense_from_parent(parent, 'prune_after10')
    from experiments.pamf_r65_four_route.stage_polyphase import StageDistilled
    return StageDistilled(parent)


def load(path, contract, roles, device='cpu'):
    item = torch.load(path, map_location='cpu', weights_only=False)
    if item.get('schema') != 'contract_matched_dura_stage_v1' or item.get('source_contract') != contract or item.get('allowed_roles') != roles:
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    if item.get('TX_labels_read') != 0 or item.get('F_reads') != 0 or item.get('target_reads') != 0:
        raise ValueError('CHECKPOINT_TARGET_CONTAMINATED')
    if item.get('epoch') != 20 or item.get('updates') != 4940:
        raise ValueError('Complete fixed-final DURA source budget required')
    model = skeleton(item['stage'])
    model.load_state_dict(item['model'], strict=True)
    model.initialization_report = item.get('initialization_report')
    return model.to(device).eval(), item


def skeleton(stage):
    from experiments.pamf_reception_restore.models import ComplexTFUNet
    from experiments.pamf_action_restore.models import ActionRestorer
    if stage == 'coarse':
        return ComplexTFUNet()
    if stage == 'action':
        return ActionRestorer(ComplexTFUNet())
    if stage == 'r17':
        from experiments.pamf_unrolled_restore.models import UnrolledRestorer
        return UnrolledRestorer('unroll_unshared', ActionRestorer(ComplexTFUNet()))
    if stage == 'r38':
        from experiments.pamf_iq_joint_compact.models import JointCompactRestorer
        return JointCompactRestorer('dense_half')
    if stage == 'r44':
        from experiments.pamf_r38_compression_methods.models import CompactSuccessor
        return CompactSuccessor('group_pruned')
    if stage == 'r46':
        from experiments.pamf_r44_pruned_refine.models import CompactSuccessor
        return CompactSuccessor('prune_refine')
    if stage == 'r65':
        from experiments.pamf_r46_delayed_prune.models import CompactSuccessor
        return CompactSuccessor('prune_after10')
    if stage == 'r69':
        from experiments.pamf_r65_four_route.stage_polyphase import StageDistilled
        return StageDistilled()
    raise ValueError('Unknown DURA stage')


def trainable(model, stage):
    if stage == 'action':
        return list(model.estimator.parameters())
    if stage == 'r17':
        return list(model.prior.parameters())
    return list(model.parameters())


def objective(model, teacher, stage, received, clean, epoch, action_arrays=None, generator=None):
    from experiments.pamf_r44_pruned_refine.models import spectral_error
    energy = clean.detach().square().mean((1, 2)).clamp_min(1e-4)
    if stage == 'coarse':
        from experiments.pamf_reception_restore.runner import pair_loss
        return pair_loss(model(received), clean)[0]
    if stage == 'action':
        from experiments.pamf_action_restore.operators import Action
        from experiments.pamf_restoration.models import complex_iq
        from experiments.pamf_action_restore.evaluation import probe_error
        _, predicted, variance = model.estimate(received)
        kernel, coefficients, noise = action_arrays
        target = Action(complex_iq(kernel), torch.complex(coefficients[:, 0], coefficients[:, 1]),
            torch.complex(coefficients[:, 2], coefficients[:, 3]))
        probes = [torch.complex(*torch.randn((2, len(clean), 256), device=clean.device, generator=generator)) / 2**.5 for _ in range(2)]
        return probe_error(predicted, target, probes) + .1 * (variance.log() - noise.log()).square()
    values = model.stage_outputs(received)[0]
    paired = (values[-1] - clean.detach()).square().mean((1, 2)) / energy
    if stage == 'r17':
        return paired
    with torch.no_grad():
        desired = teacher.stage_outputs(received)[0]
    if stage == 'r69':
        desired = desired[1:]
    if len(values) != len(desired):
        raise ValueError('Aligned source teacher trajectories required')
    fidelity = torch.stack([(a - b.detach()).square().mean((1, 2)) / energy for a, b in zip(values, desired)]).mean(0)
    weight = (10. - 9. * (epoch - 1) / 19.) if stage in ('r38', 'r44') else (1. - .9 * (epoch - 1) / 19.)
    return paired + weight * fidelity + (0. if stage in ('r38', 'r44') else .05) * spectral_error(values[-1], clean)
