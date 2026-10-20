"""User-directed fixed three-seed MEM4 extension after exposed seed01; no tuning."""
from copy import deepcopy
from pathlib import Path
from . import baseline
from experiments.cvs_f_crossbatch50 import design as prior
ROOT=Path(__file__).resolve().parents[2]
RUN='20261020-phase1-FcrossRX-MEM4-threeSeed-clean-local-m2-r01'
OWNER='codex/root/FcrossRX50-MEM4-threeSeed-local-20261020'
SEEDS=(2026092702,2026092703)
ALL_SEEDS=baseline.SEEDS
ARMS=('MEM4',)
CLASSES=prior.CLASSES
EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100;VIEWS=('clean',)
FULL_FP32_POLICY=prior.FULL_FP32_POLICY
read=prior.read;write=prior.write
REUSED_MEMORY_RUN=prior.RUN
AUTHORIZATION='要跑三种子，单种子提升难以说服'
def rows(seeds=SEEDS):
 if any(s not in SEEDS for s in seeds):raise ValueError('Only missing fixed seeds02/03; reuse completed01')
 return [dict(row_id=f'MEM4-s{s}',treatment='MEM4',model_seed=s) for s in seeds]
def config(row,base,profile,metadata):
 if row not in rows() or Path(base).name!=RUN:raise ValueError('Fixed missing MEM4 seeds02/03 in fresh output only')
 c=deepcopy(prior.config(row,str(Path(base).parent/prior.RUN),profile,metadata))
 c.update(schema='F_crossRX50_MEM4_three_seed_confirmation_v1',run_id=RUN,launch_owner=OWNER,output_root=str(Path(base)/row['row_id']/'source'),runtime_base=str(base),source_selection='Fixedfinal50; USER explicitly fixes all remaining seeds after seed01 target result; no source-promotion gate',confirmation=dict(user_request=AUTHORIZATION,reused_seed=2026092701,reused_run=REUSED_MEMORY_RUN,fixed_new_seeds=list(SEEDS),after_exposed_first_seed=True,retuning=False,independent_blind_confirmation=False))
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('runtime_base',''),c.get('runtime_profile',''),c.get('runtime_metadata','')):raise ValueError('Fixed unchanged MEM4 extension recipe differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);profile=read(c['runtime_profile']);a=baseline.make_args(c['model_seed'],device);a.wisig_pkl=profile['dataset']['manysig_pkl'];a.output_dir=c['output_root'];a.run_id=RUN;a.candidate_id=c['row_id'];a.num_workers=0;baseline.assert_recipe(a)
 if not a.a1_source_screen_only or a.eval_sat_channel or a.sat_training_mode!='disabled':raise ValueError('Source-only train, no satellite')
 return a
def fixed_authorization():
 return dict(status='SOURCE_EXPANSION_FIXED',target_access=False,source_gate_applied=False,user_authorization=AUTHORIZATION,reason='Explicit fixed three-seed extension after seed01 target exposure; old source-gate failure preserved; no tuning/selection/repeat',expand_arms=['MEM4'],fixed_new_rows=rows(),reused_seed01_run=REUSED_MEMORY_RUN,after_exposed_first_seed=True,independent_blind_confirmation=False)
