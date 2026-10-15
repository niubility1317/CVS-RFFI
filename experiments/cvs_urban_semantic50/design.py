"""Fixed dual-gain study: three fresh rows and two exact archived controls."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_urban_branch50 import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,FORMAL_VIEWS,TRUTH,CLASSES=(getattr(parent,k) for k in
    ('PROJECT','SOURCE','FULL_FP32_POLICY','FORMAL_VIEWS','TRUTH','CLASSES'))
RUN='20261015-phase1-sigtime-semantic-e50-manysig-s2026092701-m5-r01'
RELEASE='cvs_sigtime_semantic_e50_20261015_r01'
OWNER='codex/root/semantic-e50-20261015'
BASE=Path(PROJECT)/'runs'/RUN
SCENES,VIEWS,URBAN,TRAIN_SCENES=parent.SCENES,parent.VIEWS,parent.URBAN,parent.TRAIN_SCENES
ARMS=('D0','D1','D2','D3','D4')
EPOCHS,STEPS,UPDATES=50,222,11100
CALIBRATION_SEED=2026101507
read,write=parent.read,parent.write


def rows():return [dict(arm=a,model_seed=2026092701,row_id=f'{a}-s2026092701') for a in ARMS]


def fresh_rows():return [r for r in rows() if r['arm'] in ('D0','D3','D4')]


def reference_config(arm):
    if arm not in ('D1','D2'):raise ValueError('Only exact archived A0/A-J controls may be reused')
    return parent.config(next(r for r in parent.rows() if r['arm']==('A0' if arm=='D1' else 'A-J')))


def config(row):
    if row not in rows():raise ValueError('Unregistered semantic pilot row')
    c=deepcopy(parent.config(parent.rows()[0]));arm=row['arm']
    c.update(schema='sigtime_semantic_e50_v1',run_id=RUN,**row,method='cvs_urban_semantic50',launch_owner=OWNER,
        output_root=(BASE/row['row_id']/'source').as_posix(),ordinary_concat=arm!='D0',paired=False,
        labeled_generic_augmentation='disabled in D0 to match the unmodified clean component of concat controls; native U strong view unchanged' if arm=='D0' else 'native concat path bypasses generic labeled augmentor as in archived controls',
        train_scenes=[] if arm=='D0' else list(TRAIN_SCENES),curriculum=False,margin=False,urban_ssl=False,
        baseline_recipe='old gates+own EMA+cos; label33+pseudo17; fixed inner L fold; L CE throughout',
        auxiliary=dict(kind='none' if arm in ('D0','D1') else 'old_positive_direction' if arm=='D2' else 'relative_identity',
            bidirectional=arm=='D4',start_epoch=6,warmup_epochs=5,margin=.1,
            keys='stop-gradient current same-forward joint160; no independent coordinate alignment',
            reference='same-TX positive and different-TX negative share source RX/day; reference RX differs from anchor',
            quality='urban-anchor q for u->c; sqrt(q_positive*q_negative) for c->u; q=sigmoid((added-noise SNR-5)/5); suburban q=0',
            normalization='triplets mean per anchor then full labeled batch mean; D4 arithmetic half sum',
            coefficient='once from frozen A0 epoch5 source calibration; global norm target.1 each of D3/D4',
            teacher_clean_forward_in_all_rows=True,head_aux_gradient=False,gain_aux_gradient=False,
            CE_updates_all=True,extra_U_objectives=[]),
        comparison_baseline='D0 matched clean-only and D1 archived A0 ordinary concat; D2 archived A-J positive direction',
        scientific_targets=dict(clean_gain_vs_D0=True,low_urban_gain_vs_D1=True,low_urban_accuracy=.60,
            urban_gain_pp=3.,low_gain_pp=5.,clean_drop_max_pp=.5),
        source_expansion_rule='Pilot only. Any D3/D4: clean vs D0>=.005, low vs D1>=.01, urban vs D1>=0, worst urban RX vs D1>=-.01. Final50 L_dev only; no target feedback.',
        expansion='No old eight-position family expansion. Any future multi-seed semantic study uses this independent source rule.',
        reused=False,reuse_source_config=None)
    if arm in ('D1','D2'):
        original=reference_config(arm)
        c.update(reused=True,reuse_source_config=original,output_root=original['output_root'],baseline_reuse=True,
            original_row_id=original['row_id'],original_run_id=parent.RUN)
    return c


def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed semantic E50/controls configuration differs')
    return c


def make_args(c,device='cuda:0'):
    validate(c)
    if c['reused']:raise ValueError('Archived control is read-only and must not be retrained')
    a=parent.make_args(parent.config(parent.rows()[0]),device)
    a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id']
    if c['arm']=='D0':
        a.use_concat_sat_channel_aug=False;a.sat_training_mode='disabled';a.use_aug=False
    return a


def recipe():
    r=read(BASE/'semantic_calibration.json')
    if (r['status']!='SOURCE_SEMANTIC_CALIBRATED' or not r['source_only'] or r['query_read']
            or r['optimizer_updates'] or r['warmup_epoch']!=5 or r['feature_parameters']!=195041):
        raise ValueError('Need one fixed warmup source strength estimate')
    return r
