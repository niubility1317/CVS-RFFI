"""Fixed source-only ratio selection followed by three-seed mechanism controls."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_multi_action_risk import design as parent
from experiments.cvs_phase1_stack.design import args_from_validated_config
from experiments.cvs_phase1_stack.source import require_budget
from experiments.cvs_phase1_overlay.contract import read, write

ROOT=Path(__file__).resolve().parents[2]
PROJECT=parent.PROJECT; SOURCE=parent.SOURCE; CAPSULE=parent.CAPSULE; TRUTH=parent.TRUTH
CLASSES=parent.CLASSES; FULL_FP32_POLICY=parent.FULL_FP32_POLICY
SEEDS=parent.SEEDS
RUN='20261010-phase1-feature-disentangle-manysig-m48-r01'
RELEASE='cvs_feature_disentangle_20261010_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/feature-disentangle-20261010'
FIXED_ARMS=('native','random_S','S','S_raw5','S_dir5','LTR_S','LTR_S_A',
 'LTR_S_F3','LTR_S_F5','LTR_S_F10','LTR_S_AF3','LTR_S_AF5','LTR_S_AF10')
SELECTED_ARMS=('joint_noR','joint_shared','joint_direct')
ARMS=FIXED_ARMS+SELECTED_ARMS
SEARCH_ARMS=tuple('LTR_S_'+tag+str(k) for tag in ('F','AF') for k in (3,5,10))
RECIPE=deepcopy(parent.RECIPE)
RECIPE['source_u_scope']='No U access in any arm'
FEATURE_RECIPE=dict(relation_every=8,relation_weight=.1,relation_batch=48,relation_calls=5550,
 relation_packet_occurrences=266400,positive_fishr_calls=4995,
 relation_blocks=30,source_rx=[1,3,4,6,8],source_days=[0,1,2],source_classes=list(range(6)),
 style_probability=.15,style_eta=.2,style_alpha=.1,style_location='time_down',
 fishr_beta=.9,fishr_scale=30.,fishr_coordinates=960,fishr_scene_weights={'clean':1.},
 warmup_epochs=20,ramp_epochs=20,calibration_epoch=20,calibration_target_ratios=[.03,.05,.10],
 calibration_scope='mature source-L relation buckets; fixed once, no later weight adaptation',
 style_scope='native labeled clean CE only; no extra CE or KL',
 statistical_ema='gradient statistics only, not an EMA teacher',
 shared_hidden=50,independent_hidden=48,capacity_relative_tolerance=.002,
 deferred=['hard_resampling','DSU_alternative','fourth_interaction_network'])
STRESS=deepcopy(parent.STRESS)
STRESS.update(selection='three-seed source V only; clean feasibility .5pp then mean+worstRX performance; exact ties lower ratio',
              noise_tie_pp=0.)
SELECTION_RULE=dict(candidates=[3,5,10],source_arms=['LTR_S_AF3','LTR_S_AF5','LTR_S_AF10'],
 seed_count=3,baseline='LTR_S',clean_noninferiority_pp=.5,
 score='mean over seeds and all eight source views of (accuracy+worst_rx)/2',
 no_feasible='choose maximum clean score then pressure score; flag no clean-noninferior candidate; never halt for low performance',
 tie='exact score ties lower ratio',target_feedback=False,
 selected_companion='LTR_S_F{ratio}',post_selection_controls=list(SELECTED_ARMS))

def rows():
 return [dict(stage='feature_disentangle',arm=a,model_seed=s,row_id=a+'-s'+str(s)) for a in ARMS for s in SEEDS]

def feature_plan(arm,selected_ratio=None):
 p=dict(relation='none' if arm=='native' else ('random' if arm=='random_S' else 'balanced'),
        style=arm in ('LTR_S_A',*SELECTED_ARMS) or '_AF' in arm,
        fishr='none',fishr_ratio=0.,r_identity=arm!='joint_noR',selection_required=arm in SELECTED_ARMS)
 if arm=='S_raw5':p.update(fishr='raw',fishr_ratio=.05)
 elif arm=='S_dir5':p.update(fishr='direction',fishr_ratio=.05)
 elif arm in SEARCH_ARMS:p.update(fishr='direction',fishr_ratio=int(arm.split('F')[-1])/100)
 elif arm in SELECTED_ARMS:p.update(fishr='direction',fishr_ratio=selected_ratio)
 return p

def config(row,selected_ratio=None):
 arm=row['arm'];mechanism='LTR' if arm.startswith('LTR') or arm in SELECTED_ARMS else 'native'
 if arm=='joint_shared':mechanism='shared_LTR'
 if arm=='joint_direct':mechanism='random_ce'
 c=deepcopy(parent.config(dict(stage='multi_action_risk',arm=mechanism,model_seed=row['model_seed'],row_id=mechanism+'-s'+str(row['model_seed']))))
 c.update(schema='feature_disentangle_v1',run_id=RUN,**row,method='cvs_feature_disentangle',features=[],
   output_root=(BASE/row['row_id']/'source').as_posix(),risk_recipe=deepcopy(RECIPE),arm_plan=parent.plan(mechanism),
   feature_plan=feature_plan(arm,selected_ratio),feature_recipe=deepcopy(FEATURE_RECIPE),source_stress=deepcopy(STRESS),
   source_selection=deepcopy(SELECTION_RULE))
 if arm=='joint_shared':c['risk_recipe']['hidden']=50
 if arm=='joint_direct':
  # Direct IQ has the same three L/T/joint CE slots; retains the independent R
  # constraint so only the learned digital proposal model is removed.
  c['arm_plan']['paths']=['receiver']
 return c

def resolved_config(row):
 if row['arm'] not in SELECTED_ARMS:return config(row)
 selection=read(BASE/'source_selection.json')
 if selection['status']!='SOURCE_SELECTION_FROZEN' or selection['target_scores_consumed']:raise ValueError('Unfrozen source selection')
 return config(row,selection['selected_ratio'])

def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None:raise ValueError('Unknown row')
 ratio=c.get('feature_plan',{}).get('fishr_ratio')
 if r['arm'] in SELECTED_ARMS and ratio not in (.03,.05,.10):raise ValueError('Missing selected ratio')
 if config(r,ratio if r['arm'] in SELECTED_ARMS else None)!=c:raise ValueError('Changed fixed feature configuration')
 return c

def test_rows(selection=None):
 selection=read(BASE/'source_selection.json') if selection is None else selection
 if selection.get('status')!='SOURCE_SELECTION_FROZEN' or selection.get('target_scores_consumed'):
  raise ValueError('Frozen source selection required')
 ratio=int(round(selection['selected_ratio']*100))
 if ratio not in (3,5,10):raise ValueError('Invalid selected ratio')
 allowed={'LTR_S_F'+str(ratio),'LTR_S_AF'+str(ratio)}
 return [r for r in rows() if r['arm'] not in SEARCH_ARMS or r['arm'] in allowed]

def make_args(c,device='cuda:0'):
 validate(c);a=args_from_validated_config(c,device)
 a.label_smoothing=0.;a.use_unlabeled=False;a.use_ema_teacher=False
 a.lambda_u=0.;a.lambda_ent=0.;a.label_epochs=200;a.pseudo_epochs=0
 a.pseudo_domain_gate=False;a.pseudo_temporal_gate=False;a.pseudo_strong_agreement=False
 a.phase1_lr_schedule='cosine';a.phase1_lr_min=1e-6
 return a
