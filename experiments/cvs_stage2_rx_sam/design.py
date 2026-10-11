"""Prespecified three recipes x ordinary/SAM, original E200 -> final E40."""
from pathlib import Path
from copy import deepcopy
from experiments.cvs_stage2_rx_risk import design as previous
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES=(getattr(previous,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES'))
SEEDS=previous.SEEDS;FULL_FP32_POLICY=previous.FULL_FP32_POLICY
read,write=previous.read,previous.write
RUN='20261011-phase1-stage2-rx-sam-manysig-m18-r01'
RELEASE='cvs_stage2_rx_sam_20261011_r01';OWNER='codex/root/stage2-rx-sam-20261011'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('SSL','SAM_SSL','PHYS','SAM_PHYS','RANK','SAM_RANK');PRIMARY='SAM_RANK'
PAIRS=[('SAM_SSL','SSL'),('SAM_PHYS','PHYS'),('SAM_RANK','RANK'),('PHYS','SSL'),('RANK','PHYS'),('SAM_PHYS','SAM_SSL'),('SAM_RANK','SAM_PHYS')]+[(a,'P0') for a in ARMS]
learning_rate=previous.learning_rate;frequency_learning_rate=previous.frequency_learning_rate
def rows():return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Unregistered SAM continuation row')
 base=row['arm'].removeprefix('SAM_');enabled=row['arm'].startswith('SAM_')
 c=deepcopy(previous.config(next(r for r in previous.rows() if r['arm']==base and r['model_seed']==row['model_seed'])))
 c.update(schema='stage2_rx_sam_v1',run_id=RUN,**row,stage='phase1_stage2_rx_sam',method='cvs_stage2_rx_sam',method_name=row['arm'],output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
 c['sam']=dict(enabled=enabled,base_recipe=base,rho=.05 if enabled else 0.,norm='absolute L2 of trainable dense identity parameters; native Hz parameters excluded',start_update=1,identity_gradient_passes=17760 if enabled else 8880,optimizer_steps=8880,base_optimizer='unchanged AdamW',target_and_actions='computed once at unperturbed weights; detached q,mask,IQ,noise reused',dropout='replay each differentiable forward RNG; global RNG restored to first-pass end',buffers='replay from first-pass start; preserve first-pass end buffers',auxiliary='one unperturbed objective and update, never SAM-perturbed',restore='exact saved parameter copies in finally before optimizer step, including exceptions',reference='https://github.com/google-research/sam',rho_rationale='fixed initial .05, follows published example scale; no sourceV or target fitting; not claimed optimal')
 c['evaluation'].update(primary_arm=PRIMARY,control_role='three paired recipes x two optimizers, all same E200 parents/final40; equal optimizer update counts, SAM doubles gradient passes, measured cost reported')
 c['mechanism_scope']='Test robustness to a dense identity weight neighborhood while preserving unlabeled targets and source physical actions. No EMA or satellite. V health only; no source or target accuracy-based selection.'
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('SAM continuation configuration differs')
 return c
