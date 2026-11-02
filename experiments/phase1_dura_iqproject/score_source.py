"""Score the fixed source predictions against the original DURA anchor."""
from experiments.phase1_dynamic_volterra import score_source as base


def main():
    old = base.SCHEMA, base.USES_DURA
    base.SCHEMA, base.USES_DURA = 'phase1_dura_iqproject_v1', True
    try: base.main()
    finally: base.SCHEMA, base.USES_DURA = old


if __name__ == '__main__': main()
