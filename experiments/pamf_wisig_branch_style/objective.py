"""Scope branch transforms to the six existing U student forwards."""

from collections import defaultdict
from contextlib import contextmanager

from experiments.pamf_repl_completion import objectives as original
from experiments.pamf_ssl_research.hard_pl_mixstyle_objective import style_hooks
from experiments.pamf_ssl_research.hard_pl_no_as_mixstyle_objective import NoAsMixStyleObjective

from . import transforms
from .plan import ARMS


SITES = {
    'frequency_moments': ('core.id_backbone.f2', transforms.moment_mix),
    'frequency_envelope': ('core.id_backbone.f2', transforms.frequency_envelope),
    'pa_moments': ('core.id_backbone.pa_b2', transforms.moment_mix),
    'pa_envelope': ('core.id_backbone.pa_lift', transforms.pa_envelope),
    'reference_trend': ('response', transforms.reference_trend),
    **{f'signature_bend_{i}': (f'sequence_pool.views.{i}.local_projection', transforms.signature_bend)
       for i in range(3)},
}


def sites_for(branches):
    for branch in branches:
        if branch == 'signature_bend':
            yield from (f'signature_bend_{i}' for i in range(3))
        else:
            yield branch


def branch_forward(self, student, x, domain, pseudo, accepted, metadata):
    time, branches = ARMS[self.c['branch_style_arm']]
    if time and not branches:
        return original.portable_style_forward(self, student, x, domain, pseudo, accepted, metadata)

    sites = tuple(sites_for(branches))
    if not hasattr(self, 'branch_sampled_shift'):
        self.branch_sampled_shift = defaultdict(list)
    for name in sites:
        self.style_applied.setdefault(name, 0)
    first_view = self.style_calls % 6 == 0
    sample_shift = first_view and (self.style_calls // 6) % 222 == 0

    @contextmanager
    def hooks(model, prefix, recipe, partner, valid, lam, *, force, capture, enabled_by_level, prepend):
        # Draw the exact original plan, including temporal gate RNG, for every
        # arm. The middle-layer draw also gates each added branch for fair dose.
        enabled = force or enabled_by_level['t2.pool']
        time_recipe = recipe if time else dict(recipe, levels=[])
        handles = []
        applied_sites = {}
        ref = model.id_backbone.encoder
        with style_hooks(model, prefix, time_recipe, partner, valid, lam, force=force,
                         capture=capture, enabled_by_level=enabled_by_level, prepend=prepend) as applied:
            try:
                eligible = int(valid.sum()) if enabled and sites else 0
                for name in sites:
                    path, transform = SITES[name]

                    def hook(module, inputs, feature, *, name=name, transform=transform):
                        if name in applied_sites or len(feature) != len(partner):
                            raise ValueError('Branch transform must see one aligned tensor per U view')
                        applied_sites[name] = eligible
                        if not eligible:
                            return feature
                        mixed = transform(module, feature, partner, 1 - lam, valid)
                        if sample_shift:
                            difference = mixed.detach() - feature.detach()
                            rms = difference.square().mean().sqrt()
                            self.branch_sampled_shift[name].append(rms)
                        return mixed

                    handles.append(ref.get_submodule(path).register_forward_hook(hook))
                yield applied
                if set(applied_sites) != set(sites):
                    raise ValueError('A declared direct branch hook was bypassed')
                applied.update(applied_sites)
            finally:
                for handle in handles:
                    handle.remove()

    return original.portable_style_forward(self, student, x, domain, pseudo, accepted, metadata, hooks=hooks)


@contextmanager
def installed(config, holder):
    from experiments.pamf_repl_view_shuffle.source import installed as install_original

    original_method = NoAsMixStyleObjective.unlabeled_student_forward
    try:
        with install_original(config, holder, True):
            NoAsMixStyleObjective.unlabeled_student_forward = branch_forward
            yield
    finally:
        NoAsMixStyleObjective.unlabeled_student_forward = original_method
