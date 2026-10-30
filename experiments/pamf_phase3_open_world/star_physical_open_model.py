"""Read the existing C0 projector before unit normalization, in one forward."""
from __future__ import annotations

from . import star_open_world_model as oldow
from . import star_reference_open_model as reference


SEED = reference.SEED
PARAMETERS = reference.PARAMETERS
ACTIVE_PARAMETERS = reference.ACTIVE_PARAMETERS
relation = reference.relation
clear = reference.clear
parameter_report = reference.parameter_report


def install(native_model, seed=SEED):
    return reference.install(native_model, use_reference=False, seed=seed)


def forward(model, x):
    """Expose the same-call LayerNorm64 output without extra weights or detach."""
    parts = oldow._parts(model)
    if not isinstance(parts, reference.ContributionParts) or parts.use_reference is not False:
        raise ValueError('The complete constant-condition C0 architecture is required')
    captured = []

    def capture(module, inputs, output):
        if captured:
            raise ValueError('Exactly one same-call pre-unit projector output required')
        captured.append(output)

    handle = parts.projector[-1].register_forward_hook(capture)
    try:
        result = reference.forward(model, x)
        if len(captured) != 1:
            raise ValueError('The actual pre-unit LayerNorm64 output was not captured')
        raw = oldow._matrix(captured[0], (len(x), 64), 'pre-unit open_raw64')
        return dict(result, open_raw64=raw)
    except BaseException:
        clear(model)
        raise
    finally:
        handle.remove()
