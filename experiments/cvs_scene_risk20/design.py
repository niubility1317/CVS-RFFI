from copy import deepcopy
from pathlib import Path
from experiments.cvs_F_nontrue20 import design as base
read,write=base.read,base.write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-F-sceneRisk-e20-s2026092701-m2-r01'
RELEASE='cvs_F_sceneRisk_E20_20261019_r01'
LOCAL='local_artifacts/F_sceneRisk_E20_20261019';OWNER='codex/root/F-sceneRisk20'
PROJECT=read(ROOT/'runtime.json')['project'] if (ROOT/'runtime.json').is_file() else 'PROJECT'
BASE=Path(PROJECT)/'runs'/RUN;REFERENCE_RUN=base.RUN
REFERENCE_RELEASE='cvs_F_nontrue_CSIL_E20_20261019_r01';REFERENCE_COMMIT='1e7cc414f73f093dfbdccee0e4ba7cf90c7a5324'
SOURCE,FORMAL_VIEWS,TRUTH,PANEL=base.SOURCE,base.FORMAL_VIEWS,base.TRUTH,base.PANEL
SCENES,VIEWS,CLASSES,FIT,FULL_FP32_POLICY=base.SCENES,base.VIEWS,base.CLASSES,base.FIT,base.FULL_FP32_POLICY
SEED=2026092701;UPDATES=4440;STEPS=222
TRAIN_SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
model_config=base.model_config
def rows():return [dict(row_id=a+'-s'+str(SEED),arm=a,model_seed=SEED) for a in ('BALANCED','RISK')]
def reference(arm):return Path(PROJECT)/'runs'/REFERENCE_RUN/(arm+'-s'+str(SEED))
def config(row):
 if row not in rows():raise ValueError('Unregistered scene risk row')
 c=deepcopy(base.config(next(r for r in base.rows() if r['arm']=='B1')))
 c.update(schema='F_sceneRisk20_v1',run_id=RUN,**row,output_root=str(BASE/row['row_id']/'source'),launch_owner=OWNER,training_recipe='pure F ORIGINAL; scratch20=13L+7U; .5 ordinary CE +.5 scene risk CE; '+row['arm'])
 c['training_mechanism'].update(nontrue_weight=0.,teacher_packets_per_step=0,teacher_shadow=False,physical_shadow=False,teacher_target=None,teacher_correct_mask=False,extra_input='complete practical(raw)',normalization='all128 per-sample CE; group means; all present groups retain positive weight',scene_risk=dict(groups=list(TRAIN_SCENES),mode=row['arm'],eta=.01,uniform_mixture=.5,initial_logq=[0.,0.,0.],update='logq += .01 * detached current group CE; absent group increment0; subtract logsumexp; apply updated q',present_normalization='q normalized over present groups; half uniform over present groups',clean_V_for_weights=False,teacher_or_student_confidence_filter=False))
 c['source_promotion']=dict(RISK_minus_BALANCED_midlow_min_pp=.5,RISK_minus_B1_midlow_min_pp=.5,RISK_minus_B0_fullVclean_min_pp=-.2,automatic_extension=False,max_cumulative_epochs=50,decision_before_formal_scores=True,on_fail='close fixed bounded scene-risk recipe; no eta/mixture/seed rescue')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed F/source/defaulthead/E20/risk recipe differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=base.make_args(base.config(next(r for r in base.rows() if r['arm']=='B1')),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
