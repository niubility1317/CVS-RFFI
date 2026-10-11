"""Fixed E20+20, source-driven multilevel gradient and U-style factorial."""
from pathlib import Path
from copy import deepcopy
from experiments.cvs_early_geometry import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY'))
read,write,baseline=old.read,old.write,old.baseline
RUN='20261012-phase1-e20-budget-fishr-manysig-m24-r01'
RELEASE='cvs_e20_budget_fishr_20261012_r01';OWNER='codex/root/e20-budget-fishr-20261012';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('CE','SSL','MULTI','MIX_L','MIX_U','FISHR_RAW','FISHR_BUDGET','JOINT');PRIMARY='JOINT'
PAIRS=[('SSL','CE'),('MULTI','SSL'),('MIX_L','MULTI'),('MIX_U','MULTI'),('MIX_U','MIX_L'),('FISHR_RAW','MULTI'),('FISHR_BUDGET','FISHR_RAW'),('FISHR_BUDGET','MULTI'),('JOINT','MULTI'),('JOINT','MIX_U'),('JOINT','FISHR_BUDGET')]+[(a,'P0') for a in ARMS]
def rows():return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Unregistered E20 continuation row')
 a,s=row['arm'],row['model_seed'];mapped=a if a in ('CE','SSL') else 'MULTI'
 c=deepcopy(old.config(next(r for r in old.rows() if r['arm']==mapped and r['model_seed']==s)))
 c.update(schema='e20_budget_fishr_v1',run_id=RUN,**row,stage='phase1_E20_plus20',method='cvs_e20_budget_fishr',output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,model_initialization='original_CE_E20',parent_checkpoint=c['reference_checkpoint'],checkpoint_sources=[c['reference_checkpoint']])
 c['rx_action'].update(enabled=False,critic=False)
 c['losses'].update(feature=0.,margin=0.)
 c['multilevel']=dict(enabled=a not in ('CE','SSL'),mixstyle_l=a=='MIX_L',mixstyle_u=a in ('MIX_U','JOINT'),fishr_mode='raw' if a=='FISHR_RAW' else 'budget' if a in ('FISHR_BUDGET','JOINT') else 'none',interval=4,batch_size=60,aux_ce_weight=.1,fishr_raw_weight=2.,fishr_ratio=.1,fishr_cap=10000.,fishr_eps=1e-12,mixstyle_p=.5,mixstyle_alpha=.1,mixstyle_strength=.3,style_confidence=.95,levels=['time_down','t2.pool'],head_warmup_steps=200,head_warmup_lr=.003,head_warmup_weight_decay=.0001,head_warmup_batch=60,
  sampling='L only same-day 6TX x 5RX x2; U groups are detached weak predictions, both>=.95, cross RX same day',fishr_scope='actual readout weight+bias sample gradients; head values stopped in Fishr; backbone-only regularization',gradient_budget_scope='time_fuse for shallow, t2 for middle; per-term before mean; aggregate achieved norms logged')
 c['evaluation'].update(primary_arm=PRIMARY)
 c['feature_target']='none; multilevel source gradient variance and optional U main-path style'
 c['margin_target']='none'
 c['optimizer']['schedule']='reset AdamW; epoch cosine20; same LR for all identity parameters; parent20 + incremental20 = total40'
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('E20 continuation configuration differs')
 return c
def learning_rate(c,epoch):return old.learning_rate(c,epoch)
