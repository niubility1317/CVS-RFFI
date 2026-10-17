"""Two new scratch E20 rows; only actual L order differs after matched buffering."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_scene_cycle20 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261017-phase1-F-crossRX-LbatchOrder-sourceRX1-e20-s2026092701-m2-r01'
RELEASE='cvs_F_LbatchOrder_sourceRX1_e20_20261017_r01';OWNER='codex/root/F-LbatchOrder-E20-20261017'
LOCAL='local_artifacts/F_L_batch_order_E20_20261017';BASE=Path(PROJECT)/'runs'/RUN
BACKEND=old.BACKEND;BACKEND_COMMIT=old.BACKEND_COMMIT;SMOKE_SOURCE=old.SMOKE_SOURCE
SEED=2026092701;FIT=old.FIT;SCENES=old.SCENES;UPDATES=4440;ARMS=('O0','O1')
base_model_config=old.base_model_config;source_validation=old.source_validation;macro_f1=old.macro_f1

def rows():return [dict(row_id=a+'-RX1-s2026092701',arm=a,model_seed=SEED) for a in ARMS]
def legacy_config():return old.config(next(r for r in old.rows() if r['arm']=='B00'))
def config(row):
 if row not in rows():raise ValueError('Only registered matched L-order rows')
 c=deepcopy(legacy_config());c.update(schema='F_CROSS_RX_L_batch_order_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_L_batch_order',baseline_reference='Two NEW matched rows;old result cannot control buffering/private L RNG',training_recipe='Full original F+CROSS_RX/RDrop/base/8L/1.25/U;both buffer6/privateL RNG;O1only reorders within6 updates',budget_authorization='Two new scratchE20 rows,total40modelE/8880updates;no parent/reused result control',cost_claim='Each4546560LNN/originalU+EMA+V/8820held;no new NN/loss/IQ. Buffer/gather CPU time separate.')
 c['batch_order']=dict(policy='identity' if row['arm']=='O0' else 'balance_TX_RX',block_updates=6,packets_per_block=768,L_batch_size=128,blocks_per_epoch=37,epochs=20,expected_blocks=740,expected_steps=4440,order_seed=202610175101,L_sampler_seed=202610175201,L_loader_seed=202610175202,seed_rule='Fixed named private seeds;model-seed offset from2026092701,none based on source/target score',common='Both consume same private original L sequence/buffer6/compute same quotas/concat/index_select;O0identity/O1balanced',quota='For each L TX/RX cell in768packetblock distribute floor(N/6)+remainder,perbatch128;preserveevery sourceposition',L_physical_multisets='Exact original buffer multiset and all frequencies retained,no selection/redraw/deletion',U='Original13L+7U/oldgate/ownEMA;U loader/ordering untouched and never buffered',new_IQ_transform=False,new_loss_terms=0,new_NN_forwards=0,new_deploy_state=False)
 c['source_selection']=dict(primary='O1',reference='O0',fixed_final=20,held_RX=1,held_L=1260,views=['clean',*SCENES],min_clean_gain_pp=1.,min_clean_F1_gain_pp=0.,max_each_practical_drop_pp=.5,max_seenV_drop_pp=.5,final_base_L_objective_max=.5,on_fail='Closefixed6step Lreorder,no blocklength/allocation/Lseed/SupCon/CE/RX/modelseed/budget rescue',on_pass='Separatelyregistered independent source seed only;not IQAg/As/joint/formaltarget proof')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('FixedLorder/roles/budget/U/seed/weights changed')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(legacy_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a

def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('Two NEW matched rows required')
 b,s=results['O0'],results['O1'];delta={v:100*(s['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)};f1=100*(s['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);sv=100*(s['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy']);eps=1e-10
 conditions=dict(clean=delta['clean']>=1.-eps,clean_F1=f1>=-eps,each_practical=all(delta[v]>=-.5-eps for v in SCENES),seen_V=sv>=-.5-eps)
 for arm,r in results.items():
  conditions[arm+'/execution']=r['actual_steps']==4440 and r['L_forward_packets']==4546560 and r['extra_L_forward_packets']==1136640 and r['total_pair_weight']==1.25 and r['As_training_observations']==0 and r['head_fit_updates']==0 and r['target_access'] is False and r['final_base_L_objective']<=.5 and r['order_epochs']==20 and r['order_steps']==4440 and r['order_blocks']==740 and r['all_block_multisets_preserved'] and (r['moved_L_positions']==0 if arm=='O0' else r['moved_L_positions']>0)
 return dict(qualified=all(conditions.values()),conditions=conditions,delta_accuracy_pp=delta,delta_clean_F1_pp=f1,delta_seenV_pp=sv,primary='O1',source_only=True,Ag_efficacy_proved=False,As_efficacy_proved=False,joint_goal_complete=False)
