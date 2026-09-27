"""Independent named random streams, constructed only on first access."""
from collections.abc import Mapping

import numpy as np


STREAM_NAMES = ('geometry', 'states', 'shadow', 'scatter', 'phase', 'noise',
                'sync', 'tracking', 'equalizer')


class NamedRNGs(Mapping):
    """Preserve the complete mapping while deferring unused generators.

    Each generator has its original independently derived seed. Iterating names
    does not construct streams; reading values/items materializes them normally.
    Instances are per-channel, mutable through their generators, and never cached.
    """
    def __init__(self, seed, seed_deriver):
        self.seed = int(seed)
        self.seed_deriver = seed_deriver
        self._generators = {}

    def __getitem__(self, name):
        if name not in STREAM_NAMES:
            raise KeyError(name)
        if name not in self._generators:
            self._generators[name] = np.random.default_rng(self.seed_deriver(self.seed, name))
        return self._generators[name]

    def __iter__(self):
        return iter(STREAM_NAMES)

    def __len__(self):
        return len(STREAM_NAMES)

    def __contains__(self, name):
        return name in STREAM_NAMES

    @property
    def materialized_names(self):
        return tuple(name for name in STREAM_NAMES if name in self._generators)
