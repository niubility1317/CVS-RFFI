"""Teacher-qualified practical routing versus same-count-rule random routing;E30 only."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_u_practical30 import design as b
globals().update({k:v for k,v in vars(b).items() if k.isupper()})
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-FcrossRX-UQualified-e30-s2026092701-m2-r01';RELEASE='cvs_FcrossRX_UQualified_E30_20261019_r01'
OWNER='codex/root/practical-UQualified';LOCAL='local_artifacts/F_UQualified_E30_20261019_r01'
BASE=Path(PROJECT)/'runs'/RUN;COMPARATOR_RUN='20261019-phase1-FcrossRX-UPractical-e30-s2026092701-m3-r02'
COMPARATOR_CODE='fa9b272aa400ed3403e8f550c50a6957ec0f8588';ARMS=('R01','Q01');ALLOCATION_SEED=20261019731
read,write,resolved=b.read,b.write,b.resolved
_BASE_ROW=next(r for r in b.rows() if r['arm']=='C01')
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def reference_row(row):
 if row not in rows():raise ValueError('Only two fixed routing rows')
 return 'B01-s'+str(SEED)
def reference_config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Only own original As20/30 reference')
 return b.reference_config(_BASE_ROW,segment)
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only fixed original As20+10/seed01')
 c=b.config(_BASE_ROW,30);mode='QUALIFIED_PRACTICAL' if row['arm']=='Q01' else 'COUNT_RANDOM'
 c.update(schema='FcrossRX_UQualified30_v1',run_id=RUN,**row,launch_owner=OWNER,method=mode,output_root=str(BASE/row['row_id']/'source'),training_recipe='OriginalP43 L6 unchanged;common current teacher-only eligibility;rawU practical on qualified positions or private random positions of own qualified count',budget_authorization='two ownAs20+10,totalE30 each;2220new updates;matching prior N/C readouts reused',cost_claim='6L768 plusone nativeUstudent+teacher;0addedNN;AsU count dynamic actualqualification;CPU propagation recorded',baseline_reference='current matchedN/teacher-onlyC frozen from '+COMPARATOR_RUN,benchmark_context='Q-C total route increment;Q-R qualification-position versus same count rule;future teacher trajectories and therefore Ascounts may differ,reportactualdosage')
 c['u_practical'].update(mode=mode,slot_fraction=None,actual_As_fraction=None,U_As_packets=None,selected_slots=None,slot_namespace='FcrossRX/UQualified/count-routing/v1',channel_namespace='FcrossRX/train/e{epoch}/step{step}/U_QV',augmentation_seed=ALLOCATION_SEED,selection='all existing teacher base_mask positions' if row['arm']=='Q01' else 'private uniform positions, count equal to this row current teacher base_mask.sum()',selection_depends_on_current_teacher_qualification=True,selection_uses_TX_or_pseudo_class=False,teacher_trajectory_equality_claim=False,same_future_Ascount_claim=False,unqualified_IQ='Gaussian inQ;randomAs possibleinR',entropy_normalization='unchanged mean over allU, includinganyunqualifiedAs inR',L_changed=False)
 c['source_promotion']=dict(fullV_Q_minus_N_clean_min_pp=-.1,panel_midlow_Q_minus_C_min_pp=.5,source_failure_skips_formal=False,automatic_extension=False,on_fail='close fixed qualified-routing;noE50/ratio/threshold/seed rescue',Q_minus_R='attribution comparison only;not an extra promotion gate')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed qualified route/data/parent/seed/budget differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=b.make_args(b.config(_BASE_ROW),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 return a
