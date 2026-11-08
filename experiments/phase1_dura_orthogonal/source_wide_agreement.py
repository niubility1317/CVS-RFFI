"""Compare frozen direct wide-range models with the clean-source reader proxy."""
from experiments.phase1_dura_orthogonal import source_frozen_agreement as base

CANDIDATES = {
    'direct_narrow': ('20261108-phase1-direct-large-timefreq-stable-source-s2026092701-r01',
                      '4bc1aa7209c1ad7ac305f2802ebc6dee195c207a'),
    'direct_wide': ('20261108-phase1-direct-timefreq-wide-residual-source-s2026092701-r01',
                    'dc48d330da8b95a2e7de7a0eac7ba4bc2e142a82'),
    'direct_wide_physics': ('20261108-phase1-direct-wide-physics-source-s2026092701-r01',
                            '1a10d53c0c35d1613300f0c0e413d96ccdd800e2'),
}


def main():
    base.CANDIDATES = CANDIDATES
    base.main()


if __name__ == '__main__':
    main()
