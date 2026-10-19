from copy import deepcopy
from pathlib import Path
from experiments.cvs_F_nontrue20 import design as base
read,write=base.read,base.write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-F-conditionalTail-e20-s2026092701-m2-r01'
RELEASE='cvs_F_conditionalTail_E20_20261019_r01'
LOCAL='local_artifacts/F_conditionalTail_E20_20261019';OWNER='codex/root/F-conditionalTail20'
PROJECT=read(ROOT/'runtime.json')['project'] if (ROOT/'runtime.json').is_file() else 'PROJECT'
BASE=Path(PROJECT)/'runs'/RUN;REFERENCE_RUN=base.RUN
REFERENCE_RELEASE='cvs_F_nontrue_CSIL_E20_20261019_r01';REFERENCE_COMMIT='1e7cc414f73f093dfbdccee0e4ba7cf90c7a5324'
SOURCE,FORMAL_VIEWS,TRUTH,PANEL=base.SOURCE,base.FORMAL_VIEWS,base.TRUTH,base.PANEL
SCENES,VIEWS,CLASSES,FIT,FULL_FP32_POLICY=base.SCENES,base.VIEWS,base.CLASSES,base.FIT,base.FULL_FP32_POLICY
SEED=2026092701;UPDATES=4440;STEPS=222;SECOND_DROPOUT_SEED=20261019631
model_config=base.model_config

def rows():return [dict(row_id=a+'-s'+str(SEED),arm=a,model_seed=SEED) for a in ('CE2','TAIL2')]

def reference(arm):return Path(PROJECT)/'runs'/REFERENCE_RUN/(arm+'-s'+str(SEED))

def config(row):
 if row not in rows():raise ValueError('Unregistered conditional channel risk row')
 c=deepcopy(base.config(next(r for r in base.rows() if r['arm']=='B1')))
 c.update(schema='F_conditionalTail20_v1',run_id=RUN,**row,output_root=str(BASE/row['row_id']/'source'),launch_owner=OWNER,training_recipe='pure F ORIGINAL; scratch20=13L+7U; .5ordinary CE+.5two-draw conditional As risk; '+row['arm'],budget_authorization='User fixed F scratch cumulative E20; no automatic extension; source-qualified own20+30 requires matched controls/seams',cost_claim='CE2 and TAIL2 same 3L/384 student packets per step, same complete physical draws, no external teacher, no extra VJP; distinct student backward cost from reused B1')
 c['training_mechanism']=dict(mode='ORIGINAL',alpha=0.,CROSS_RX=False,RDrop=False,L_forwards_per_sample=3,ordinary_CE_weight=.5,extra_CE_weight=.5,extra_input='two independent complete practical(raw) observations of every same legal L packet',CE_temperature=1.,CE_smoothing=.01,teacher_packets_per_step=0,teacher_shadow=False,teacher_target=None,conditional_risk=dict(mode=row['arm'],draws=2,CE2='mean of two CE for each packet, then mean over128',TAIL2='.5*mean(two CE)+.5*max(two CE), independently within each physical packet; mean over128',detached_weights='CE2 .5/.5; TAIL2 .75/.25 to larger/smaller CE, exact ties .5/.5',positive_all_As=True,per_packet_weight_sum=1.,student_confidence_gate=False,physical_draw_0='unchanged native train namespace/step',physical_draw_1='same actual epoch; mirrored namespace step=4441-native_step; native seeds/config/session/complete engine unchanged',second_dropout='private torch fork seed=20261019631+native_step; restores global native RNG',same_receiver_hardware_session='RX/day/eq; never TX',scene_prior='original iid uniform low_suburban/mid_urban/low_urban; independent complete draws',V_or_target_weight_fit=False),U='original hidden TX gate+ownEMA unchanged',new_deploy_parameters=0)
 c['source_promotion']=dict(TAIL2_minus_CE2_midlow_min_pp=.5,TAIL2_minus_B1_midlow_min_pp=.5,TAIL2_minus_B0_fullVclean_min_pp=-.2,automatic_extension=False,max_cumulative_epochs=50,decision_before_formal_scores=True,on_fail='close fixed two-draw half-mean half-max conditional risk; no mixture/K/seed rescue')
 return c

def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Fixed F/defaulthead/full source/E20/two-draw recipe differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=base.make_args(base.config(next(r for r in base.rows() if r['arm']=='B1')),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
