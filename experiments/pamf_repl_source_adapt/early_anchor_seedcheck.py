"""Verify the source-selected E20 correction on two independent seeds."""
import argparse
import sys

from experiments.cric_dura_comparison.io import read, write
from . import early_label_anchor as base, early_label_anchor_register as registration
from .early_anchor_reach import registered


RUN = '20261121-phase1-pamf-tri-label01-earlyanchor-seedcheck-clean-m2-r01'
MODULE = 'experiments.pamf_repl_source_adapt.early_anchor_seedcheck'


def rows():
    return [dict(row_id=f'labeled_1pct-tri-e20-repair60-s{seed}', setting='labeled_1pct',
                 protocol_id='label01', model_seed=seed, weight=6., views=['clean'],
                 reuses_original=False, mode='replace', floor=.55)
            for seed in (2026092702,2026092703)]


def register():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['register'])
    p.add_argument('--template',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    d=registration.make(read(a.template))
    d.update(display_name='1% 标注 E20 替换的两个新种子复核',
             description='固定源域优先的 0.60 配方，验证两个未参与候选设计的模型种子。')
    d['execution']['local_artifact_root']='local_artifacts/pamf_tri_label01_earlyanchor_seedcheck_20261121'
    d['permissions']['claim_scope']='Two training seeds held out from recipe development, historical clean target'
    d['expected_artifacts']=['2 same-seed real E20 smokes', '2 independent E50 clean and U scores', '2 same-ID contrasts']
    for entry,row in zip(d['rows'],rows(),strict=True):
        entry['seeds']['model']=row['model_seed']
    write(a.output,d)


if __name__=='__main__':
    with registered(RUN,MODULE,rows):
        if sys.argv[1:2]==['register']:
            register()
        else:
            base.main()
