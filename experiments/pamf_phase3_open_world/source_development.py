"""Four-class scratch PAMF for source-only held-out-class development."""
from __future__ import annotations

import importlib
from pathlib import Path
from types import MethodType


ORDER = (0, 5, 3, 6, 2, 1, 4, 7)
FROZEN_KEYS = {'F', 'AF', 'BASE', 'FOLD', 'G'}


def _validate(runtime, model_seed, class_count):
    if not isinstance(runtime, dict) or set(runtime) != {'base_release', 'frozen_sources'}:
        raise ValueError('Only immutable code roots are accepted; checkpoints and data inputs are forbidden')
    if not isinstance(runtime['base_release'], str) or not runtime['base_release']:
        raise ValueError('Immutable base code release is required')
    roots = runtime['frozen_sources']
    if (not isinstance(roots, dict) or set(roots) != FROZEN_KEYS
            or any(not isinstance(value, str) or not value for value in roots.values())):
        raise ValueError('Exact immutable dependency code roots are required')
    if isinstance(model_seed, bool) or not isinstance(model_seed, int) or model_seed <= 0:
        raise ValueError('Positive integer scratch model seed required')
    if isinstance(class_count, bool) or not isinstance(class_count, int) or class_count != 4:
        raise ValueError('This source development diagnostic requires exactly four classes')


def _activate_code(runtime):
    base = Path(runtime['base_release']) / 'experiments'
    if not base.is_dir():
        raise ValueError('Immutable base experiment code is missing')
    package = importlib.import_module('experiments')
    package.__path__ = [str(base)] + [str(path) for path in package.__path__ if Path(path) != base]
    bootstrap = importlib.import_module('experiments.cric_dura_comparison.bootstrap')
    bootstrap.activate(runtime)
    bootstrap.numerical_policy()
    current = str(Path(__file__).resolve().parents[1])
    package.__path__ = list(dict.fromkeys([current, str(base)] + list(package.__path__)))


def build_model(runtime, model_seed, class_count=4):
    """Build on CPU from RNG only, retaining the native bias-free cosine head.

    The legacy ReferenceCVS constructor first allocates a random six-class head.
    Its class weight is replaced by a fresh four-class weight before any forward;
    no trained state is loaded, inherited, partially reused or frozen.
    """
    _validate(runtime, model_seed, class_count)
    import torch
    from torch import nn

    # Keep initialization reproducible without consuming the trainer's RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(model_seed)
        _activate_code(runtime)
        from experiments.cvs_reference_identity.model import ReferenceCVS
        from experiments.cvs_rdrop_sinc_units.x3_features import CompositePool, CompositeReference
        from experiments.cvs_rdrop_sinc_units.coordinate import CoordinateSinc
        from experiments.cric_dura_comparison.supervised_physics import shuffled_descriptor
        from experiments.cvs_source_identity_readout.input_policy import apply

        reader = ReferenceCVS('reference_response')
        core = reader.core.id_backbone
        # Preserve the original head class, norms, physical fusion and scale 30.
        # There is no class bias; reducing 6 to 4 removes exactly 2*160 parameters.
        core.cls_head.weight = nn.Parameter(torch.empty(class_count, 160))
        nn.init.xavier_uniform_(core.cls_head.weight)
        apply(reader)
        core.sinc = CoordinateSinc(core.sinc, 'CENTER_25K')
        reader.sequence_pool = CompositePool(core.t2.pw.out_channels, 'SIG_TIME_X3', model_seed)
        reader.__class__ = CompositeReference
        for view in reader.sequence_pool.views:
            view.register_buffer('window_order', torch.tensor(ORDER), persistent=False)
            view.forward = MethodType(shuffled_descriptor, view)

        if (sum(parameter.numel() for parameter in reader.parameters()) != 232673
                or tuple(core.cls_head.weight.shape) != (4, 160)
                or tuple(reader.sequence_pool.output_projection.weight.shape) != (160, 336)
                or len(reader.sequence_pool.views) != 3
                or any(view.window_order.tolist() != list(ORDER) for view in reader.sequence_pool.views)
                or hasattr(core.cls_head, 'bias') or core.cls_head.scale != 30.):
            raise ValueError('Four-class current PAMF identity structure differs')
        if any(not parameter.requires_grad or parameter.device.type != 'cpu' for parameter in reader.parameters()):
            raise ValueError('All identity branches and default head must be jointly trainable on CPU at construction')

    reader.source_development_contract = dict(schema='pamf_phase3_source_development_model_v1',
        initialization='scratch_only', model_seed=model_seed, class_count=class_count,
        parameters=232673, feature_dimension=160, input_shape=[2, 256],
        signature_views=3, temporal_order=list(ORDER), sinc_coordinates='CENTER_25K',
        default_head='original_bias_free_cosine_scale30', default_head_joint=True,
        unused_initial_six_class_weight='random_only_replaced_before_first_forward',
        checkpoint_reads=0, source_reads=0, target_reads=0, teacher_networks=0,
        unused_dual_networks=0, training_objective='caller_supplied_labeled_source_CE_only',
        evidence_scope='scratch model construction; not source development or target performance')
    return reader
