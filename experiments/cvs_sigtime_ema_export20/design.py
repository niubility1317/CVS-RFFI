"""Reuse the fixed all-five-RX BASE_SSL E20 own EMA, without new training."""
from pathlib import Path
from experiments.cvs_evidence_responsibility_train20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,key) for key in ('PROJECT','SOURCE','CAPSULE','TRUTH','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-SIGTIME-ownEMA-export-clean-manysig-m3-r01'
RELEASE='cvs_SIGTIME_ownEMA_export_clean_20261015_r01'
DEPENDENCY_RELEASE=PROJECT+'/releases/'+parent.RELEASE
BASE=Path(PROJECT)/'runs'/RUN
VIEWS=('clean',);READOUTS=('NATIVE_HEAD',)
GATE=dict(first_seed_only=SEEDS[0],EMA_source_V_errors_not_higher=True,EMA_source_V_CE_not_higher=True,
    at_least_one_strict_improvement=True,CE_strict_tolerance=1e-8,
    outcome='If qualified freeze all three fixed own EMA exports, predict clean, then independent truth-last scoring; otherwise source-only closure',
    no_decay_window_parent_seed_scan=True)

def rows():return [dict(row_id='EMA-s'+str(seed),arm='EMA_EXPORT',model_seed=seed) for seed in SEEDS]
def parent_config(seed):return parent.config(next(row for row in parent.rows() if row['arm']=='BASE_SSL' and row['model_seed']==seed))
def config(row):
    if row not in rows():raise ValueError('Only all three fixed BASE_SSL own EMA rows are registered')
    c=parent_config(row['model_seed'])
    return dict(c,schema='phase1_SIGTIME_ownEMA_export_v1',run_id=RUN,**row,
        launch_owner='codex/root/SIGTIME-ownEMA-export-20261015',output_root=(BASE/row['row_id']/'source').as_posix(),
        baseline_reuse=False,stage='fixed_ownEMA_export_evaluation',
        budget_authorization='User same-trajectory export priority; reuse all three fixed all-source BASE_SSL parents, first-seed source gate before any target; no new training',
        cost_claim='Zero new backbone epochs/optimizer steps; existing parents3x20/4440; source V27000 unique reads and54002 forwards per row;3x168000 new clean decisions and3x168000 reused student decisions',
        baseline_definition='All three canonical actualLS0 BASE_SSL seeds from the fixed original M12; chosen by role/seed rather than target score',
        checkpoint_parent_run=parent.RUN,checkpoint_parent_row=c['row_id'],checkpoint_state_key='ema_model',
        checkpoint_training_data='Original full five-RX L6300/U56700hidden/V27000, exact parent source contract',
        training_mechanism=dict(new_optimizer_steps=0,new_backbone_epochs=0,existing_parent_epochs=20,existing_parent_updates=4440,
            export='Own original final EMA including all feature/head/gain/norm state; no cross-seed averaging, no teacher update or head fitting',
            EMA_decay=.999,sinc_coordinates='CENTER_25K',target_feedback=False),
        source_falsification=GATE,goal=dict(clean_three_seed_mean=.84,goal_completed=False,
            innovation_claim=False,role='Independent performance export measure, never rescue a failed LOCKED mechanism'))
def validate(c):
    row=next((row for row in rows() if row['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed same-trajectory own EMA export differs')
    return c

def qualify(student,ema):
    if student['count']!=27000 or ema['count']!=27000:raise ValueError('Complete original source V required')
    finite=all(__import__('math').isfinite(row['mean_CE']) for row in (student,ema))
    if not finite:raise ValueError('Finite source CE required')
    return (ema['errors']<=student['errors'] and ema['mean_CE']<=student['mean_CE'] and
        (ema['errors']<student['errors'] or ema['mean_CE']<student['mean_CE']-GATE['CE_strict_tolerance']))
