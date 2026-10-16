"""Fixed real same-TX coherent IQ interpolation; source-only E20 falsification."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_additive_as20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-FcrossRX-coherentmix-source-RX1-e20-s2026092702-m3-r02';RELEASE='cvs_FcrossRX_coherentmix_source_E20_20261016_r02';OWNER='codex/root/FcrossRX-coherentmix-20261016'
BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_F_crossRX_additiveAs_confirm_sourceRX1_e20_20261016_r01'
BACKEND_COMMIT='05984a1229ebac7025a95081dc5fae4108276b28'
REFERENCE_RUN='20261016-phase1-F-crossRX-additiveAs-confirm-sourceRX1-e20-s2026092702-m3-r01'
SMOKE_SOURCE=Path(PROJECT)/'runs'/REFERENCE_RUN/'D_raw-RX1-s2026092702/source'
SEED=2026092702;HELD=1;RX=(1,3,4,6,8);FIT=(3,4,6,8);STEPS=222;UPDATES=4440;ARMS=('SAME_RX','CROSS_RX');SCENES=parent.SCENES;PAIR_SEED=20261016961
macro_f1=parent.macro_f1

def template():
 if parent.SEED!=SEED or parent.base_model_config()['model_seed']!=SEED:raise ValueError('Pinned model-seed profile must activate before config/model construction')
 return parent.config(next(r for r in parent.rows() if r['arm']=='D_raw'))
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Only two new source coherent-interpolation rows')
 c=deepcopy(template());c.update(schema='F_crossRX_coherent_IQ_interpolation_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_crossRX_coherent_IQ_interpolation',training_recipe='Unchanged8Lforward base plus .25R(coherent(real sameTX/day partner,raw));original U/defaulthead;no As')
 c['training_mechanism'].update(Ag=True,As=False,extra_pair_input='bounded coherent '+row['arm']+' interpolation of raw',raw_total_weight=.75)
 c['additive'].update(extra_input=c['training_mechanism']['extra_pair_input'],operator='One source IQ interpolation;no As,ordinary-before-mix,extra loss or U view')
 c['coherent_mix']=dict(mode=row['arm'],partner='Current filtered fitL only;same trueTX/day,distinct physicalpacket;SAME_RX or uniform other3fitRX',reference_window=[80,160],complex_scalar='LS donor-to-anchor on fixed reference;not deconvolution or TX inversion',mixture_fraction=.25,normalization='Retain anchor full-packet L2 norm',max_relative_L2_change=.10,projection='Shortest real-vector sphere arc from anchor to normalized coherent mixture;all I/Q samples same scalar coefficients',label_condition='Partner selection only;all formulas/hyperparameters common to all TX',pair_seed=PAIR_SEED,refresh='Private seed+step;does not advance native random streams',all_L_supervised=True,U_or_V_donors=False,new_physical_records=False,hardware_replacement_claim=False)
 c['source_selection']=dict(primary='CROSS_RX',references=['reused_D_raw','SAME_RX'],fixed_final=20,held_RX=1,held_L=1260,min_clean_gain_vs_raw_pp=1.,min_clean_gain_vs_same_pp=.5,min_clean_macroF1_gain_pp=0.,standalone_practical_drop_reporting_threshold_pp=.5,standalone_practical_is_nonblocking=True,max_seenV_drop_pp=.5,max_final_common_base_objective=.5,numerical_tolerance_pp=1e-10,on_fail='Close fixed coherent interpolation;no fraction/bound/window/RX/seed/budget rescue',on_pass='Only supports independent source confirmation;not Ag efficacy across seeds,joint goal or target')
 c.update(baseline_reference=SMOKE_SOURCE.as_posix(),baseline_reuse=False,budget_authorization='Active Ag/As goal;two new scratchE20(40modelE/8880updates) plus exact matching existing seed D_raw;no inherited weights',cost_claim='Each new4546560LNN+originalU/EMA/V/8820held;568320mix+extra partner exposures;5040fitL bank reads;reference cost separately;dispatcher2syntheticCPsmoke')
 return c

def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Fixed coherent interpolation config/data/budget differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=parent.make_args(template(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 if a.seed!=SEED or (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing)!=(20,13,7,.01) or a.source_fold_held_rx!=HELD or a.baseline_ckpt or a.teacher_ckpt:raise ValueError('Exact scratchE20/model-seed source contract required')
 return a
source_validation=parent.source_validation

def qualify(results,reference):
 if set(results)!=set(ARMS) or reference['arm']!='D_raw' or reference.get('target_access') or reference.get('U_truth_read') or reference.get('checkpoint_sources'):raise ValueError('Complete source-only matching controls required')
 s=results['CROSS_RX'];conditions={};comparisons={};eps=1e-10
 for name,b in [('reused_D_raw',reference),('SAME_RX',results['SAME_RX'])]:
  delta={v:100*(s['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)};f1=100*(s['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);dv=100*(s['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy']);need=1. if name=='reused_D_raw' else .5
  checks=dict(clean=delta['clean']>=need-eps,clean_macroF1=f1>=-eps,seen_V=dv>=-.5-eps)
  conditions.update({name+'_'+k:v for k,v in checks.items()});comparisons[name]=dict(all_view_gain_pp=delta,clean_macroF1_gain_pp=f1,seen_V_gain_pp=dv,conditions=checks,standalone_practical_protection=all(delta[v]>=-.5-eps for v in SCENES))
 conditions['healthy']=all(r['actual_steps']==4440 and r['L_forward_packets']==4546560 and r['supcon_applied']==3330 and r['final_base_L_objective']<=.5 and r['mix']['calls']==4440 and r['mix']['packets']==568320 and r['mix']['unique_partner_packets']==5040 and r['mix']['max_relative_L2_change']<=.10001 and r['As_training_observations']==0 and r['teacher_mix_calls']==0 and not r['head_fit_updates'] and not r['target_access'] and not r['U_truth_read'] and not r['checkpoint_sources'] and r['held_RX']==HELD and r['fit_RX']==list(FIT) for r in results.values())
 return dict(qualified=all(conditions.values()),conditions=conditions,comparisons=comparisons,source_only=True,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False)
