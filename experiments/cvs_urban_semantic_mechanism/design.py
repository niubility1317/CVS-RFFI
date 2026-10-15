from pathlib import Path
from experiments.cvs_urban_semantic50 import design as training

ROOT,PROJECT,SOURCE=training.ROOT,training.PROJECT,training.SOURCE
RUN='20261015-phase1-sigtime-semantic-source-manysig-s2026092701-m5-r01'
RELEASE='cvs_sigtime_semantic_source_20261015_r01'
OWNER='codex/root/semantic-source-20261015'
BASE=Path(PROJECT)/'runs'/RUN
FORWARD_PACKETS=36722
read,write=training.read,training.write


def eligible_rows():
    records=[];reference=None
    for arm in ('D1','D0','D2','D3','D4'):
        row=next(r for r in training.rows() if r['arm']==arm);c=training.config(row)
        actual=c['reuse_source_config'] if c['reused'] else c;source=Path(c['output_root'])
        done=read(source/'completion.json');frozen=read(source/'source_frozen.json');inner=read(source/'inner_source_contract.json')
        if (done['config']!=actual or frozen['config']!=actual or done['epoch']!=50 or done['optimizer_steps']!=11100
                or done['target_access'] or done['head_fit_updates'] or frozen['selection']!='final_only'
                or frozen['target_access'] or frozen['epoch']!=50):
            raise ValueError('All five original default-head source final50 freezes required')
        if reference is None:reference=inner
        if inner!=reference:raise ValueError('Actual physical L_train/L_dev folds differ')
        records.append((row,c))
    return records,reference
