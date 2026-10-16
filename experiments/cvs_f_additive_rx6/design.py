"""Independent held-RX6 confirmation of the fixed source-PASS additive recipe."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_additive_as20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-FcrossRX-additiveAs-source-RX6-e20-s2026092701-m3-r01'
RELEASE='cvs_FcrossRX_additiveAs_RX6_source_E20_20261016_r01'
OWNER='codex/root/FcrossRX-additiveAs-RX6-20261016'
BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_F_crossRX_additiveAs_sourceRX1_e20_20261016_r01'
BACKEND_COMMIT='7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b'
SMOKE_SOURCE=Path(PROJECT)/'runs/20261016-phase1-FcrossRX-practical-source-confirm-s2026092701-m8-r01/C0-RX6-s2026092701/source'
HELD=6;RX=(1,3,4,6,8);FIT=(1,3,4,8);SEED=2026092701;STEPS=222;UPDATES=4440;ARMS=parent.ARMS;SCENES=parent.SCENES
macro_f1=parent.macro_f1

def rows():return [dict(row_id=f'{a}-RX6-s{SEED}',arm=a,model_seed=SEED) for a in ARMS]
def template(arm='D00'):return parent.config(next(r for r in parent.rows() if r['arm']==arm))
def config(row):
 if row not in rows():raise ValueError('Only three fixed heldRX6 source rows permitted')
 c=deepcopy(template(row['arm']));fold=deepcopy(c['fold']);fold.pop('all_model_teacher_training_calibration_and_feature_cache_inputs_exclude_RX1',None)
 fold.update(held_RX=HELD,training_RX=list(FIT),all_training_calibration_probe_feature_cache_inputs_exclude_held_RX=True)
 c.update(schema='F_CROSS_RX_additive_As_RX6_confirmation_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),stage='independent_source_RX6_confirmation',fold=fold,
  source_confirmation=dict(discovery_run=parent.RUN,question='Does fixed additive organization protect sourceRX6 where earlier different pairedAs lost clean?',held_RX=HELD,model_seed=SEED,no_seed02_RX1_duplication=True,
   on_fail='Stop fixed additive candidate expansion;no coefficient/RX/scene/seed/threshold/epoch rescue. Do not stop other healthy-owner jobs.',on_pass='Only sourceRX6 single-seed evidence;not Ag efficacy,crossseed,target or jointgoal'),
  budget_authorization='Active Ag+As goal,E20 permodel;three new scratch sourceRX6 controls,60modelE/13320updates,0inheritedweights/no target',
  cost_claim='Each unchanged8Lforward E20/4440/4546560LNN plus originalU/ownEMA/V/8820held readouts;As568320 onlyD_As,ordinary568320each;realCPsmoke2 separate,unusedfitL144 reads as original.')
 c['source_selection']['held_RX']=HELD
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed RX6 roles,budget,weights or source controls differ')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=parent.make_args(template(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.source_fold_held_rx=HELD;a.source_fold_training_receivers=','.join(map(str,FIT))
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing)!=(20,13,7,.01) or a.baseline_ckpt or a.teacher_ckpt:raise ValueError('ScratchE20 original U/defaulthead required')
 return a

def source_validation(model,loader,device):
 import torch
 model.eval();hits=total=0;groups={}
 with torch.no_grad():
  for x,y,domain,meta in loader:
   correct=(model(x.to(device)).argmax(1).cpu()==y)
   for rx in meta['rx_i'].unique().tolist():
    mask=meta['rx_i']==rx;v=groups.setdefault(str(rx),[0,0]);v[0]+=int(correct[mask].sum());v[1]+=int(mask.sum())
   hits+=int(correct.sum());total+=len(y)
 if total!=21600 or set(groups)!=set(map(str,FIT)):raise ValueError('Source health V must exclude heldRX6')
 rates={k:v[0]/v[1] for k,v in groups.items()};return dict(source_val_count=total,source_val_accuracy=hits/total,source_val_rx_accuracy=rates,source_val_worst_rx=min(rates.values()),held_RX_excluded=True)

def qualify(results):
 if set(results)!=set(ARMS) or any(r.get('held_RX')!=HELD or r.get('fit_RX')!=list(FIT) or r.get('U_truth_read') or r.get('checkpoint_sources') for r in results.values()):raise ValueError('All legal matching RX6 results required')
 q=parent.qualify(results);eps=1e-10
 for ref,part in q['comparisons'].items():
  delta=part['all_view_gain_pp'];protect=0. if ref=='D00' else -.5
  checks=dict(clean=delta['clean']>=protect-eps,clean_macroF1=part['clean_macroF1_gain_pp']>=protect-eps,low_urban=delta[SCENES[-1]]>=5.-eps,urban_mean=part['urban_mean_gain_pp']>=3.-eps,practical_protection=all(delta[v]>=-.5-eps for v in SCENES),seen_V=part['seen_V_gain_pp']>=-.5-eps)
  part['conditions']=checks;q['conditions'].update({ref+'_'+k:v for k,v in checks.items()})
 q.update(qualified=all(q['conditions'].values()),held_RX=HELD,model_seed=SEED,source_only=True,crossseed_confirmed=False,automatic_target=False,goal_complete=False,numerical_tolerance_pp=eps);return q
