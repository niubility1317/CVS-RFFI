import argparse
from .io import read


def run(rt):
    from .bootstrap import activate,numerical_policy
    activate(rt);numerical_policy()
    from .stage2_smoke import run as smoke
    from .continuation_setup import prepare
    from .pipeline import run as dispatch
    smoke(rt)
    prepare(rt)
    dispatch(rt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    a=p.parse_args();run(read(a.config))
