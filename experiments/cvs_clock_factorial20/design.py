"""Fixed sample-clock factorial, full source and unchanged practical observations."""
from copy import deepcopy
import json
from pathlib import Path
from experiments.cvs_fullsource_concat20 import design as ref
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-phase1-F-crossRX-clock-factorial-e20-s2026092701-m3-r01'
RELEASE='cvs_F_clock_factorial_e20_20261017_r01'
LOCAL='local_artifacts/F_clock_factorial_20261017'
OWNER='codex/root/F-clock-factorial-20261017'
read,write=ref.read,ref.write
SOURCE,CAPSULE,TRUTH,VIEWS_ROOT=ref.SOURCE,ref.CAPSULE,ref.TRUTH,ref.VIEWS_ROOT
PROJECT,AF_RELEASE=ref.PROJECT,ref.AF_RELEASE
SEED,UPDATES,STEPS,FIT=ref.SEED,4440,222,ref.FIT
CLASSES,FULL_FP32_POLICY,SCENES,VIEWS=ref.CLASSES,ref.FULL_FP32_POLICY,ref.SCENES,ref.VIEWS
BASE=Path(PROJECT)/'runs'/RUN
SMOKE_SOURCE=ref.BASE/('REPEAT-s'+str(SEED))/'source'
CLOCK=dict(max_ppm=80.,anchor_sample=127.5,half_width=16,boundary='reflect',seed=202610175411)
ARMS=('B00','B10','B01','B11')
def runtime():
 p=ROOT/'runtime.json'
 return read(p if p.is_file() else ROOT/LOCAL/'runtime.json')
def rows():return [dict(row_id=a+'-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def train_rows():return [r for r in rows() if r['arm']!='B00']
def reference_config(arm='REPEAT'):return ref.config(next(r for r in ref.rows() if r['arm']==arm))
def config(row):
 if row not in rows():raise ValueError('Only the fixed single-seed clock factorial is registered')
 c=deepcopy(reference_config());a=row['arm'];reuse=a=='B00'
 c.update(schema='F_clock_factorial20_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=SMOKE_SOURCE.as_posix() if reuse else (BASE/row['row_id']/'source').as_posix(),baseline_reuse=reuse,method='bounded_clock_factorial',stage='fullsource_clock_factorial',source_selection='Fixedfinal20/allrows;originalVhealthonly/no sourceRXholdout',baseline_reference='Existing fixed REPEAT is exact B00;existing full CONCAT is additional mandatory reference',new_Ag=a in ('B10','B11'),As=a in ('B01','B11'))
 c['clock']=dict(CLOCK)
 c['factorial']=dict(Ag=c['new_Ag'],As=c['As'],partition='(batch_position+global_step)%2;G even/S odd;64each;no labels/quality',Ag_input='SRO(shared generated ordinary)',As_input='unchanged complete practical(raw before ordinary)',absent_slot='same generated ordinary',common_pair_weight=.5,extra_pair_weight=.5,operator_marginal_packet_fraction=.5,composition_view=False)
 c['training_mechanism'].update(L_forwards_per_sample=4,expected_L_forward_packets=2273280,pair_weights=[.5,.5],second_pair='one mixed128 view,64Gslot+64Sslot',same_IQ_within_RDpair=True,ordinary_before_satellite=False,all_satellite_L_supervised=True,new_U_augmentation=False,new_crossview_loss=False,objective='.5RD(ordinary128)+.5RD(mixed128)+originalSupCon(ordinary)',hook_correction='Registered L objective minus native smoothedCE1')
 c['goal'].update(Ag_clean_gain_vs_B00=True,B11_clean_gain_vs_B00=True,B11_low_gain_vs_B01_and_fullAs=True)
 c['training_recipe']='Original full-source F+CROSS_RX/RDrop1/LS.01/oldU;four matched L forwards;'+a
 c['budget_authorization']='3 new scratchE20/60modelE/13320updates;B00 and fullAs are frozen existing references,not new training or parents'
 c['cost_claim']='Eachnewmodel2273280L forwards;G/As284160observationsperenabledfactor;originalU/teacher/test/smoke measured separately'
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Clock/data/budget/head/U/view contract differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Never retrain or initialize from the frozen B00 reference')
 a=ref.make_args(reference_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
base_model_config=ref.base_model_config
