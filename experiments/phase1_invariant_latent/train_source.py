"""Train paired source views with a phase-invariant latent objective."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_invariant_latent.model import InvariantLatentRecovery
from experiments.phase1_invariant_latent.objective import latent_loss
from experiments.phase1_volterra_aligned.objective import source_loss

_current = [None]


def make_model():
    _current[0] = InvariantLatentRecovery()
    return _current[0]


def paired_auxiliary(received, output, groups):
    if _current[0] is None or _current[0].last_latent is None:
        raise RuntimeError('Current model latent required')
    loss, _ = latent_loss(_current[0].last_latent, groups)
    return loss


def main():
    names = ('DynamicVolterra', 'SCHEMA', 'LOG_PREFIX', 'source_loss',
             'auxiliary_source_objective', 'PAIRED_VIEWS', 'USES_DURA')
    old = tuple(getattr(base, name) for name in names)
    new = (make_model, 'phase1_invariant_latent_v1', 'INVARIANT_LATENT',
           source_loss, paired_auxiliary, True, False)
    for name, value in zip(names, new):
        setattr(base, name, value)
    try:
        base.main()
    finally:
        for name, value in zip(names, old):
            setattr(base, name, value)
        _current[0] = None


if __name__ == '__main__':
    main()
