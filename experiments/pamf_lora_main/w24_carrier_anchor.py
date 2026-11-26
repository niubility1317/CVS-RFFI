"""Anchor the final likelihood at the observed dechirped carrier."""

from functools import partial
import importlib

from experiments.pamf_lora_main.w24_final_radius import projected_stages


def final_map(archive, action, y, current, prior, variance, remaining, _noise):
    # For the received carrier c, use c + A(x - c) as the forward model.
    # The affine offset changes the innovation but leaves the CG normal matrix intact.
    carrier = y.mean(dim=-1, keepdim=True).expand_as(y)
    offset = action(carrier) - carrier
    return archive.map_two(action, y + offset, current, prior, variance,
                           action.adjoint((remaining + offset) / variance))


def restorer(training, *, anchored):
    archive = importlib.import_module('.dura_capacity', training.DuraW24.__module__.rpartition('.')[0])
    solver = partial(final_map, archive) if anchored else None

    class CarrierAnchor(training.DuraW24):
        def stages(self, y, quality, rms, *, condition=()):
            return projected_stages(self, archive, y, quality, rms, condition=condition,
                                    expand_radius=False, final_solver=solver)

    return CarrierAnchor()
