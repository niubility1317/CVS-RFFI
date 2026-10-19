from copy import deepcopy
from pathlib import Path
from experiments.cvs_F_nontrue20 import design as base
read,write=base.read,base.write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-F-capabilityCE-e20-s2026092701-m2-r01'
RELEASE='cvs_F_capabilityCE_E20_20261019_r01'
LOCAL='local_artifacts/F_capability_CE_E20_20261019';OWNER='codex/root/F-capability-CE20'
PROJECT=read(ROOT/'runtime.json')['project'] if (ROOT/'runtime.json').is_file() else 'PROJECT'
BASE=Path(PROJECT)/'runs'/RUN;REFERENCE_RUN=base.RUN
REFERENCE_RELEASE='cvs_F_nontrue_CSIL_E20_20261019_r01';REFERENCE_COMMIT='1e7cc414f73f093dfbdccee0e4ba7cf90c7a5324'
SOURCE,FORMAL_VIEWS,TRUTH,PANEL=base.SOURCE,base.FORMAL_VIEWS,base.TRUTH,base.PANEL
SCENES,VIEWS,CLASSES,FIT,FULL_FP32_POLICY=base.SCENES,base.VIEWS,base.CLASSES,base.FIT,base.FULL_FP32_POLICY
SEED=2026092701;UPDATES=4440;STEPS=222;MASK_SEED=20261019621
model_config=base.model_config
def rows():return [dict(row_id=a+'-s'+str(SEED),arm=a,model_seed=SEED) for a in ('SHUFFLED','CAPABILITY')]
def reference(arm):return Path(PROJECT)/'runs'/REFERENCE_RUN/(arm+'-s'+str(SEED))
def config(row):
 if row not in rows():raise ValueError('Unregistered capability CE row')
 c=deepcopy(base.config(next(r for r in base.rows() if r['arm']=='B1')))
 c.update(schema='F_capability_CE20_v1',run_id=RUN,**row,output_root=str(BASE/row['row_id']/'source'),launch_owner=OWNER,training_recipe='pure F ORIGINAL; scratch20=13L+7U; .5ordinary CE+.5 normalized capability-weighted As CE; '+row['arm'])
 c['training_mechanism']=dict(mode='ORIGINAL',alpha=0.,CROSS_RX=False,RDrop=False,L_forwards_per_sample=2,ordinary_CE_weight=.5,extra_CE_weight=.5,extra_input='complete practical(raw)',CE_temperature=1.,CE_smoothing=.01,teacher_packets_per_step=128,teacher_shadow=False,teacher_target='only argmax==legal L TX bit; no posterior/feature/derivative target',teacher='existing verified frozen CSIL last200; no student inheritance',capability=dict(mode=row['arm'],uniform_fraction=.5,weight='w_i=.5+.5*bit_i/mean(bit); all-zero bit fallback ones',normalization='mean of all128 weights exactly1; CE denominator all128',random_control='permute teacher-correct bit within each true TX x source RX; same positive count per group',private_mask_seed=MASK_SEED,private_step_seed='MASK_SEED+optimizer_step',student_confidence_use=False,V_fit_or_weight_update=False),U='original hidden TX gate+ownEMA unchanged',new_deploy_parameters=0)
 c['source_promotion']=dict(CAPABILITY_minus_SHUFFLED_midlow_min_pp=.5,CAPABILITY_minus_B1_midlow_min_pp=.5,CAPABILITY_minus_B0_fullVclean_min_pp=-.2,automatic_extension=False,max_cumulative_epochs=50,decision_before_formal_scores=True,on_fail='close fixed capability-label-CE allocation; no mixture/teacher/seed rescue')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed F/source/defaulthead/E20/capability recipe differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=base.make_args(base.config(next(r for r in base.rows() if r['arm']=='B1')),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
