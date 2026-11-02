"""Train an anchored correction using paired-view latent consistency."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_crossview_latent.model import CrossViewDURA
from experiments.phase1_dura_volterra.pipeline import load_components, input_noise, predict_batch
from experiments.phase1_invariant_latent.objective import latent_loss
from experiments.phase1_volterra_aligned.objective import source_loss

_current = [None]


def make_model():
    _current[0] = CrossViewDURA()
    return _current[0]


def paired_auxiliary(received, output, groups):
    if _current[0] is None or _current[0].last_latent is None:
        raise RuntimeError('Current model latent required')
    return latent_loss(_current[0].last_latent, groups)[0]


def main():
    names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_loss',
             'auxiliary_source_objective', 'PAIRED_VIEWS', 'USES_DURA',
             'load_components', 'input_noise', 'predict_batch', 'extra_frozen_components')
    old = tuple(getattr(base, name) for name in names)
    values = (make_model, 'phase1_dura_crossview_latent_v1', 'DURA_CROSSVIEW_LATENT',
              source_loss, paired_auxiliary, True, True, load_components,
              input_noise, predict_batch, lambda dura, eye: (dura,))
    for name, value in zip(names, values):
        setattr(base, name, value)
    try:
        base.main()
    finally:
        for name, value in zip(names, old):
            setattr(base, name, value)
        _current[0] = None


if __name__ == '__main__':
    main()
