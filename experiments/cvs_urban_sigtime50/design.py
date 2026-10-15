"""Single-seed four-row E50 pilot authorized after the completed E30 study."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_urban_sigtime import design as previous

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY=(getattr(previous,k) for k in
    ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY'))
RUN='20261015-phase1-sigtime-urban-ssl-e50-manysig-s2026092701-m4-r01'
RELEASE='cvs_sigtime_urban_ssl_e50_20261015_r01'
OWNER='codex/root/urban-sigtime-e50-20261015'
BASE=Path(PROJECT)/'runs'/RUN
FORMAL_VIEWS=previous.FORMAL_VIEWS
SCENES,VIEWS,URBAN,TRAIN_SCENES=(getattr(previous,k) for k in ('SCENES','VIEWS','URBAN','TRAIN_SCENES'))
SEEDS=(2026092701,)
ARMS=('B0','C3','SSL_CONF','SSL_WEAK')
PRIMARY='SSL_CONF'
EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,STEPS,UPDATES=50,33,17,222,11100
read,write,diagnostic_config=previous.read,previous.write,previous.diagnostic_config


def rows():
    return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]


def config(row):
    if row not in rows():raise ValueError('Unregistered E50 row')
    base='B0' if row['arm']=='B0' else 'C3' if row['arm']=='C3' else 'C4'
    c=deepcopy(previous.config(next(r for r in previous.rows() if r['arm']==base and r['model_seed']==row['model_seed'])))
    c.update(schema='sigtime_urban_ssl_e50_v1',run_id=RUN,**row,method='cvs_urban_sigtime50',
        launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),
        parent_training_epochs=0,incremental_epochs=50,total_epochs=50,epochs=50,
        label_epochs=33,pseudo_epochs=17,updates=11100,
        expansion='Four registered single-seed rows only; no automatic seed expansion',
        checkpoint_selection='final_only',baseline_recipe='old gates+own EMA+cos; label33+pseudo17; L CE throughout')
    mode='clean_confidence' if row['arm']=='SSL_CONF' else 'same_sample_weak_agreement' if row['arm']=='SSL_WEAK' else 'disabled'
    c['ssl_plan'].update(acceptance=mode,confidence_threshold=.95,weak_std=.01,weak_seed=2026101501,
        source_calibration='Own EMA on full legal V at first pseudo step; precision>=.95 and coverage>=.1',
        native_old_gates_unchanged=True,urban_student_confidence_gate=False)
    c['source_evidence']='Previous source V own-EMA calibration: confidence .95 precision 99.82-99.87%, coverage 90.97-91.95%; no target-based parameter choice'
    return c


def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Frozen scratch E50 pilot configuration differs')
    return c


def make_args(c,device='cuda:0'):
    validate(c)
    base='B0' if c['arm']=='B0' else 'C3' if c['arm']=='C3' else 'C4'
    old=previous.config(next(r for r in previous.rows() if r['arm']==base and r['model_seed']==c['model_seed']))
    a=previous.make_args(old,device)
    a.epochs=50;a.label_epochs=33;a.pseudo_epochs=17
    a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id']
    a.checkpoint_selection='final_only';a.test_eval_start_epoch=999999
    return a
