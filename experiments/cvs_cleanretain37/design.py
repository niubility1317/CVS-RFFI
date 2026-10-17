"""User-directed clean-retaining As organization;one scratch30 row, no new Ag."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_cycleorder_factorial32 import design as ref
from experiments.cvs_fullsource_concat30_25 import design as ref25
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,TRUTH,CLASSES,SMOKE_SOURCE,FORMAL_VIEWS,SCENES,VIEWS,TRAIN_SCENES,SEED,STEPS,UPDATES,RX=(getattr(ref,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','TRUTH','CLASSES','SMOKE_SOURCE','FORMAL_VIEWS','SCENES','VIEWS','TRAIN_SCENES','SEED','STEPS','UPDATES','RX'))
RUN='20261017-phase1-FcrossRX-cleanretain-e30-s2026092701-m1-r01';RELEASE='cvs_FcrossRX_cleanretain_E30_20261017_r01';OWNER='codex/root/cleanretain37';BASE=Path(PROJECT)/'runs'/RUN
parent_config=ref.parent_config
ARMS=('C0','AS50','R25')
def rows():return [dict(arm='R25',row_id='R25-s'+str(SEED),model_seed=SEED)]
def all_rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def source_row_root(arm):
 if arm=='R25':return BASE/rows()[0]['row_id']
 if arm=='C0':return ref25.BASE/('B00-s'+str(SEED))
 if arm=='AS50':return ref.BASE/('B01-s'+str(SEED))
 raise ValueError('Only declared actual references')
def control_config(arm):
 if arm=='C0':return ref25.config(next(r for r in ref25.rows() if r['arm']=='B00'))
 if arm=='AS50':return ref.config(next(r for r in ref.rows() if r['arm']=='B01'))
 raise ValueError('Only unchanged existing controls')
def row_config(row):return config(row) if row['arm']=='R25' else control_config(row['arm'])
def config(row):
 if row not in rows():raise ValueError('One full-source R25 row only')
 c=deepcopy(control_config('AS50'));c.pop('cycle_order');c.update(schema='FcrossRX_cleanretain_scratch30_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),Ag=False,As=True,training_recipe='primary ordinary128;secondary ordinary64+fullpractical(raw64);two identical generated-IQ RDrop forwards;each256halvesCE normalized .5/.5;SupCon primary;20L10U scratch30')
 c['training_mechanism'].update(primary='unchanged ordinary(x)',secondary='ordinary on alternating64;original full As(raw) on remaining64',clean_repeat_control=False,ordinary_supervision_weight=.75,satellite_supervision_weight=.25,satellite_generated_per_L_batch=64,all_generated_satellite_supervised=True,paired_clean_raw_control=False,new_losses=False)
 c['channel']['input']='raw normalized physical L before ordinary;exact native As for selected64;no new physical perturbations'
 c['clean_retention']=dict(mask='(batch_position+global_step)%2==1;exact64/128;no TX/quality dependence',preserved_primary=True,ordinary_slots='192/256 effective physical views',satellite_slots='64/256',source_reference_chat='01a1427e-49c0-7840-a84d-5bb38d5a8164',recipe_reference='E20+10 B01 clean-retaining organization;no old weights or clock/FIR reused',new_Ag=False,new_consistency_loss=False,new_U_augmentation=False)
 c['source_qualification']=dict(rule='fullVclean>=C0-.25pp;each2drawx2urban>=AS50-2pp;source-only descriptive protection,not overall goal proof',automatic_seed_extension=False,formal_all_new_rows=True)
 c['benchmark_context']='Human directed promising exposed-benchmark training organization;not independent blind method selection. No target-selected checkpoint is loaded;all new training scratch30.'
 c['source_basis']=dict(question='Does retaining ordinary supervision preserve identity while actual nominal practical training keeps urban adaptation under scratch30?',matching='exactP25C0/P32AS50 actual args;native losses/U/RDrop/network/epochs unchanged',old_frozen_filter_clock_timing='negative;not reused',no_source_holdout=True)
 return c

def validate(c):
 if c!=config(rows()[0]):raise ValueError('Fixed scratch30/fullsource/defaulthead/.75+.25 contract differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=ref.make_args(control_config('AS50'),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
