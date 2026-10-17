"""First fixed-filter training question;old source feasibility failure remains unchanged."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_clock_factorial30 import design as trained
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-phase1-F-crossRX-spectralIQ-firsttrain-e20plus10-s2026092701-m2-r01'
RELEASE='cvs_F_spectralIQ_firsttrain_E30_20261017_r01'
LOCAL='local_artifacts/F_spectralIQ_firsttrain_E30_20261017'
OWNER='codex/root/F-spectralIQ-firsttrain-E30'
read,write=trained.read,trained.write
PROJECT,SOURCE,CAPSULE,TRUTH,VIEWS_ROOT=trained.PROJECT,trained.SOURCE,trained.CAPSULE,trained.TRUTH,trained.VIEWS_ROOT
AF_RELEASE,SEED,FIT,SCENES,VIEWS,CLASSES,FULL_FP32_POLICY=trained.AF_RELEASE,trained.SEED,trained.FIT,trained.SCENES,trained.VIEWS,trained.CLASSES,trained.FULL_FP32_POLICY
TRAIN_SCENES,CHANNEL_SEEDS=trained.TRAIN_SCENES,trained.CHANNEL_SEEDS
STEPS,UPDATES=222,2220
BASE=Path(PROJECT)/'runs'/RUN
CONTROL_RUN='20261017-phase1-F-crossRX-clock-e30-score-recovery-s2026092701-m5-r01'
CONTROL_ROOT=Path(PROJECT)/'runs'/CONTROL_RUN
BANK_RUN='20261016-diagnostic-F-crossRX-spectralIQ-fitL-s2026092701-m1-r01'
BANK_ROOT=Path(PROJECT)/'runs'/BANK_RUN
BANK_PATH=BANK_ROOT/'kernel_bank.json'
FILTER_SEED=20261017991
CONTROL_ARM={'B10':'B00','B11':'B01'}
def runtime():
 p=ROOT/'runtime.json';return read(p if p.is_file() else ROOT/LOCAL/'runtime.json')
def rows():return [dict(row_id='FIR_'+a+'-s'+str(SEED),arm=a,model_seed=SEED) for a in CONTROL_ARM]
def train_rows():return rows()
def nominal_config(arm):return trained.config(next(r for r in trained.rows() if r['arm']==CONTROL_ARM[arm]))
def parent_config(arm):return trained.parent_config(CONTROL_ARM[arm])
def parent_source(arm):return trained.parent_source(CONTROL_ARM[arm])
def parent_proof(arm):return trained.parent_proof(CONTROL_ARM[arm])
def validate_parent_checkpoint(cp,c):return trained.validate_parent_checkpoint(cp,dict(c,arm=CONTROL_ARM[c['arm']]))
base_model_config=trained.base_model_config

def config(row):
 if row not in rows():raise ValueError('Only first fixed FIR B10/B11 training rows')
 c=deepcopy(nominal_config(row['arm']))
 c.update(run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),schema='F_FIXED_FIR_FIRST_TRAIN_E20PLUS10_V1',new_Ag=True,As=row['arm']=='B11',clock=None,model_initialization='strict own nonAg B00/B01 sourcefinal20+ownEMA;resetAdamW/cos10',parent_selection='Factor-defined noAg sourcefinal20 parents;not target-selected',training_recipe='Original4L/commonRD.5+extraRD.5/SupCon.1/U;fixedFIR only on64Ag slots in E21..30',filter_policy=dict(bank_run=BANK_RUN,bank='All12originalstructuredkernels uniformly;no kernel/range/refit/subset change',bank_fit_data='Original L5040 subset of current originalL6300;not model weights,all5RX trained',source_feasibility_result='FAIL_FIXED_FILTER_CLOSED remains unchanged:finitePSD vscircularoracle threshold failed;source RX association/identity passed;training utility was never tested',question='First empirical training utility of unchanged bounded digital observation FIR,not exact RX replacement',taps=33,pre_RMS_response_bound_dB=.5,post_RMS_peak_to_peak_bound_dB=1.,boundary='reflect16/centered/remove16numericdelay',normalization='Restore original packetRMS',selection_seed=FILTER_SEED,label_dependent_application=False,input='existing ordinary Ag slots only;shared ordinary unchanged,As separatelyraw',TX_PA_IQ_CFO_noise_added=False),control_run=CONTROL_RUN,control_row=CONTROL_ARM[row['arm']]+'-s'+str(SEED),new_neural_forwards=0,new_losses=0,new_U_rules=0,budget_authorization='2firsttrainingrows;each sameE20parent+10 cumulativeE30/2220newupdates',selection='fixedfinal10/cumulative30,no targetfeedback,allrows finish',scientific_falsification=dict(B10_clean_vs_B00_min_pp=.2,B11_clean_vs_B00_min_pp=0.,B11_low_vs_B01_min_pp=.5,B11_mid_vs_B01_min_pp=0.,B11_each_suburban_vs_B01_min_pp=-.25,no_automatic_seed_extension=True))
 c['method']='F_fixed_spectral_IQ_first_training'
 c.update(stage='fullsource_fixed_FIR_first_training',checkpoint_sources=[(parent_source(row['arm'])/'final_ssdg.pth').as_posix()],source_selection='Fixedfinal10/cumulative30;Vhealthonly,no trainingRXholdout',checkpoint_selection='final_only_segment10_cumulative30',cost_claim='Each child2220newupdates/6660cumulative;1136640LNN,142080Ag,0or142080As;sourceguard163804NN separately;originalU/EMA/eval/test measured separately',baseline_reference='Exact factor-mapped nonAg B00/B01 E30 controls;FULL_AS larger Asdose descriptive reference',baseline_reuse=False)
 c['goal']=dict(clean_net_positive=True,low_urban_above=.6,each_suburban_gap_exploratory_pp=2,target_feedback=False,joint_goal_complete=False)
 c['factorial']=dict(arm=row['arm'],Ag='FixedFIR on64 existingordinary slots,E21..30 only',As='Originalcomplete practical on64raw slots' if c['As'] else 'None',common_supervision='.5 originalordinary RD',extra_supervision='.5 meanextraRD;64Ag plus64As/control slots',total_L_scale=1.,L_forwards=4,composed_As_Ag_view=False)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('FixedFIR/fullsource/E30/parent contract differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=trained.make_args(nominal_config(c['arm']),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
