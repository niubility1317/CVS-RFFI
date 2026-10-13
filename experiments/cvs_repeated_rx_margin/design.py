from copy import deepcopy
from pathlib import Path
from experiments.cvs_report_alignment_clean import design as b
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write,RX=(getattr(b,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write','RX'))
RUN='20261013-phase1-repeated-rx-margin-clean-manysig-m12-r01';RELEASE='cvs_repeated_rx_margin_20261013_r01';OWNER='codex/root/repeated-rx-margin-20261013';BASE=Path(PROJECT)/'runs'/RUN;COUNT=168000;VIEWS=('clean',);ARMS=('MATCHED_BASE','RANDOM_ROBUST','MEASURED_MEAN','MEASURED_ROBUST');DAYS=(1,2,3)
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for a in ARMS for s in SEEDS]
def readouts(c):return ('FINAL',)
def config(row):
 if row not in rows():raise ValueError('Unregistered repeated drift row')
 c=deepcopy(b.config(next(r for r in b.rows() if r['arm']=='TX_CONDITIONAL' and r['model_seed']==row['model_seed'])))
 c.pop('alignment');c.update(run_id=RUN,**row,output=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,classifier=dict(kind='ordinary_cos',scale=30.,dimension=160,RX_heads=False,routing=False,margin=False,Phase2='discardW refit legal support; no drift banks retained'),drift=dict(fit_role='L_only',cells=[6,5,3],folds=3,assignment='eachTX/RX/day sortedphysicalIDs rank modulo3',refresh='currentmodel eval/no_grad everyepoch; restore modes/RNG/state',normalization_eps=1e-4,q_eps=1e-12,radius_max=.2,lambda_weight=.25,warmup_epochs=5,ramp_epochs=10,direction='equalnorm random' if row['arm']=='RANDOM_ROBUST' else 'measured crossfit',objective='base' if row['arm']=='MATCHED_BASE' else 'mean CE' if row['arm']=='MEASURED_MEAN' else 'percompetitor worst margin',random_seed='primary+epoch*104729; independentGenerator',gradient_diagnostics='first and last optimizer step each epoch; netadded versus CE+SSL; encoder and cosW',persistent_query_state=False),budget_authorization='user report 20261013 fourarms threeprimaryseeds fromscratch30, 12models79920updates;30x6300 statisticforward/model extra cost disclosed, noVgate/cleanonly',matched_control='MATCHED_BASE same primary seed',goal=dict(clean_three_seed_mean=.84,feedback_to_training=False))
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Repeated drift fixedrecipe differs')
 return c
def strength(epoch):return min(1.,max(0.,(epoch-5)/10))
