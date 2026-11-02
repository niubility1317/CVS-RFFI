"""Apply truth-last source scoring to the fixed cross-view model."""
from experiments.phase1_dynamic_volterra import score_source as base


def main():
    old = base.SCHEMA; base.SCHEMA = 'phase1_spectral_crossview_v1'
    try: base.main()
    finally: base.SCHEMA = old


if __name__ == '__main__': main()
