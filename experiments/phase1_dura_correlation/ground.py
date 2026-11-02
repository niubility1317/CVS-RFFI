"""Fixed correlation DURA source replay and truth-last Phase1 scoring."""

from importlib import import_module
from experiments.phase1_dura_correlation.integrated import CorrelationPosteriorInDura

SOURCE = '20261102-phase1-dura-correlation-source-s2026092701-r01'


def main():
    base = import_module('experiments.phase1_dura_posterior.ground')
    old_model, old_source, old_nonlocal = base.PosteriorInDura, base.SOURCE, base.NONLOCAL_SOURCE
    base.PosteriorInDura = CorrelationPosteriorInDura
    base.SOURCE = SOURCE
    base.NONLOCAL_SOURCE = SOURCE
    try:
        base.main()
    finally:
        base.PosteriorInDura, base.SOURCE, base.NONLOCAL_SOURCE = old_model, old_source, old_nonlocal


if __name__ == '__main__':
    main()
