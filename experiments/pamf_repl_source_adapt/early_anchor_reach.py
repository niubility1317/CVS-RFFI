"""Test two source-calibrated confidence levels of the original anchor."""
import argparse
from contextlib import contextmanager

from experiments.cric_dura_comparison.io import read, write
from . import early_label_anchor as base, early_label_anchor_register as registration


RUN = '20261121-phase1-pamf-tri-label01-earlyanchor-reach-clean-m2-r01'
MODULE = 'experiments.pamf_repl_source_adapt.early_anchor_reach'


def rows():
    return [dict(row_id=f'labeled_1pct-tri-e20-repair{name}-s{base.SEED}', setting='labeled_1pct',
                 protocol_id='label01', model_seed=base.SEED, weight=6., views=['clean'],
                 reuses_original=False, mode='replace', floor=floor)
            for name,floor in (('75',.70),('60',.55))]


@contextmanager
def registered(run=RUN,module=MODULE,candidates=rows):
    old=base.RUN,base.MODULE,base.rows
    base.RUN,base.MODULE,base.rows=run,module,candidates
    try:
        yield
    finally:
        base.RUN,base.MODULE,base.rows=old


def register():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['register'])
    p.add_argument('--template',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    d=registration.make(read(a.template))
    d.update(display_name='Tri 低标注 E20 校验范围',description='扩大 E20 替换覆盖的两档固定源域门槛。')
    d['execution']['local_artifact_root']='local_artifacts/pamf_tri_label01_earlyanchor_reach_20261121'
    d['expected_artifacts']=['2 real E20 smokes, E50 frozen clean predictions, U scores and paired results']
    d['notes'].append('Thresholds 0.75 and 0.60 were predefined from the source V disagreement readout.')
    write(a.output,d)


if __name__=='__main__':
    import sys
    with registered():
        if sys.argv[1:2]==['register']:
            register()
        else:
            base.main()
