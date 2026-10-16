"""Two scratch E20 arms change only ordinary-view coupling across physical RX."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_crossrx_satmix20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,SCENES,source_validation,macro_f1=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','SCENES','source_validation','macro_f1'))
RUN='20261017-phase1-FcrossRX-rxgroup-sourceRX1-e20-s2026092701-m2-r01';RELEASE='cvs_FcrossRX_rxgroup_sourceRX1_E20_20261017_r01';OWNER='codex/root/FcrossRX-rxgroup-20261017'
BASE=Path(PROJECT)/'runs'/RUN;BACKEND=PROJECT+'/releases/cvs_F_crossRX_additiveAs_sourceRX1_e20_20261016_r01';BACKEND_COMMIT='7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b'
SEED=2026092701;FIT=(3,4,6,8);HELD=1;UPDATES=4440;ARMS=('BATCH','RXGROUP')
SMOKE_SOURCE=Path(PROJECT)/'runs'/parent.RUN/'B00-RX1-s2026092701/source'
def template():return parent.config(next(r for r in parent.rows() if r['arm']=='B00'))
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Two registered source coupling arms only')
 c=deepcopy(template());c.update(schema='F_CROSS_RX_ordinary_RX_coupling_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_ordinary_coupling',updates=UPDATES,budget_authorization='Active Ag/As goal;two matched scratch E20 models,4440 updates each,0 inherited states;source-only RX1 falsification',baseline_reuse=False,baseline_reference='New matched BATCH scratchE20;original B00 state used compatibility-smoke only',source_falsification=None)
 c['training_mechanism'].update(Ag='coupling_candidate_not_validated' if row['arm']=='RXGROUP' else False,As=False)
 c['coupling']=dict(mode=row['arm'],scope='Only labeled normal ordinary anti-shortcut and general-channel stages before original normalPA/DAC',grouping='physical RX only,never TX/day/quality/loss',seed=20261017831,refresh='private independent step/stage/RX stream;no physical hardware replacement claim',single_packet_operators='Exact native code/probabilities/ranges/order/boundaries/RMS policy unchanged',shared_original='PA/DAC/class-signature and all native U paths unchanged;common global RNG consumption retained',shadow='Both arms compute whole-batch and allRX-group stage outputs;return only registered mode,no additional NN',supcon='Existing first/second same-IQ ordinary maps;no new loss/projection/head or coefficient',formal_test_changed=False)
 c['training_recipe']='Original sixL .25RD(ordinary)+.75RD(raw),same originalSupCon/U/defaulthead;change ordinary batch coupling only,noAs'
 c['source_selection']=dict(primary='RXGROUP',reference='BATCH',fixed_final=20,held_RX=1,held_L=1260,min_clean_gain_pp=1.,min_clean_macroF1_gain_pp=0.,max_seenV_drop_pp=.5,max_final_L_CE=.5,all_practical_reported_but_nonblocking=True,on_fail='Close fixed RX-group ordinary coupling;no operator subset/seed/RX/grouping/threshold/budget rescue',on_pass='Only separately registered secondseed source confirmation,not Ag/joint/target proof')
 c['cost_claim']='Two scratchE20/8880updates;each3409920LNN plusoriginalU/EMA/V and8820held=expected4686678rowNN;dispatcherCPsmoke2 separate;two extra IQ stage shadows but0extraNN'
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Fixed current source/coupling/budget differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=parent.make_args(template(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.seed,a.label_smoothing)!=(20,13,7,SEED,.01) or a.baseline_ckpt or a.teacher_ckpt or not a.use_aug or a.use_concat_sat_channel_aug:raise ValueError('Original scratch E20/U/ordinary contract required')
 return a
def qualify(results,reference=None):
 if set(results)!=set(ARMS):raise ValueError('Both new matched arms required')
 b,r=results['BATCH'],results['RXGROUP'];delta={v:100*(r['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)};f1=100*(r['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);v=100*(r['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy']);eps=1e-10
 checks=dict(clean=delta['clean']>=1.-eps,macroF1=f1>=-eps,seenV=v>=-.5-eps,healthy=all(x['actual_steps']==4440 and x['L_forward_packets']==3409920 and x['supcon_applied']==3330 and x['final_L_CE']<=.5 and x['coupling']['calls']==4440 and x['coupling']['stages']==8880 and x['coupling']['packets']==568320 and x['coupling']['coverage_cells']==72 and not x['As_training_observations'] and not x['target_access'] and not x['U_truth_read'] and not x['checkpoint_sources'] and not x['head_fit_updates'] for x in results.values()))
 return dict(qualified=all(checks.values()),checks=checks,all_view_gain_pp=delta,clean_macroF1_gain_pp=f1,seenV_gain_pp=v,urban_mean_gain_pp=sum(delta[s] for s in SCENES[3:])/3,source_only=True,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False)
