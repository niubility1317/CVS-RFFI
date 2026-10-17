"""Fixed descriptive historical-control audit;no training/candidate gate."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN='20261017-diagnostic-FcrossRX-matchedRAW-sourceRX1-s2026092701-m5-r01';OWNER='codex/root/FcrossRX-RAWcontrol-audit-20261017'
PROJECT='/home/szu2070436088/2510044040/CV-SincNet';SEED=2026092701
REFS=(('refresh_RAW','20261017-phase1-FcrossRX-viewrefresh-sourceRX1-e20-s2026092701-m3-r01','RAW'),('timing_RAW','20261017-phase1-FcrossRX-AsTiming-sourceRX1-e20-s2026092701-m4-r01','RAW'),('cycle_B00','20261017-phase1-F-crossRX-sceneCycle-sourceRX1-e20-s2026092701-m3-r01','B00'),('awgn_B00','20261017-phase1-F-crossRX-isolatedAWGN-sourceRX1-e20-s2026092701-m3-r01','B00'),('phase_B00','20261017-phase1-F-crossRX-boundedPhase-sourceRX1-e20-s2026092701-m3-r01','B00'))
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
FILES=('resolved_config.json','source_contract.json','initialization.json','completion.json','source_frozen.json','actual_native_args.json','actual_channel.json','actual_modules.json','ordinary_augmentation_actual.json','satmix_result.json','training_sequence.jsonl.gz','satmix_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz','epoch_metrics.jsonl','metrics_epoch.jsonl','epoch_metrics.csv','metrics_epoch.csv','initial_smoke.pt')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sources():return [dict(key=k,run_id=r,arm=a,source=PROJECT+'/runs/'+r+'/'+a+'-RX1-s2026092701/source') for k,r,a in REFS]
def config():return dict(schema='FcrossRX_five_nominalRAW_controls_descriptive_v1',run_id=RUN,owner=OWNER,sources=sources(),views=list(VIEWS),new_model_epochs=0,new_optimizer_updates=0,new_network_forwards=0,target_access=False,source_only=True,source_scope='One exposed heldsourceRX1L1260,modelseed2026092701,5historicalcontrols not5seeds',weights='Only initial_smoke.pt untrained random initialstates,never final trainedCP;CPU tensorcomparison only',data='No IQ/Utruth/target access;only complete frozen source predictions and permitted L/U opaqueID metadata',candidate_promotion=False,limits='Descriptive conditional recipe/roles/init/sequence/stored-input-scalars comparison;no exactIQ replay or uniqueCUDAcausalclaim;no closedroute rescue or current scientificcriteria modification')
def validate(c):
 if c!=config():raise ValueError('Fixed five sourcecontrols/no target/training overrides')
 return c

def verify_source(base,ref):
 base=Path(base);done=read(base/'completion.json');init=read(base/'initialization.json');freeze=read(base/'source_frozen.json');contract=read(base/'source_contract.json');c=read(base/'resolved_config.json');args=read(base/'actual_native_args.json');r=read(base/'satmix_result.json')
 if ref not in sources() or c['run_id']!=ref['run_id'] or c['arm']!=ref['arm'] or c['model_seed']!=SEED or c['output_root']!=ref['source']:raise ValueError('Only fixed known source-control provenance')
 if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['target_access'] or done['target_evaluated'] or done['head_fit_updates'] or not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access'] or freeze['checkpoint']!=ref['source']+'/final_ssdg.pth':raise ValueError('Known scratch source-only FINAL20 history required before initial snapshot read')
 if contract['held_RX']!=1 or contract['source_receivers']!=[3,4,6,8] or contract['native_role_comparison']!='EXACT_MATCH_CARRIER_THEN_SOURCE_FOLD' or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Matching fourRX source roles required,no target-contaminated states')
 roles=contract['role_ids']
 if any(len(v)!=len(set(v)) for v in roles.values()) or set(roles['L_s'])&(set(roles['U_s'])|set(roles['V'])) or set(roles['U_s'])&set(roles['V']):raise ValueError('Source roles must be separate')
 if (args['seed'],args['epochs'],args['label_epochs'],args['pseudo_epochs'],args['source_fold_held_rx'])!=(SEED,20,13,7,1) or args['baseline_ckpt'] or args['teacher_ckpt'] or args['use_concat_sat_channel_aug'] or not args['use_aug']:raise ValueError('Original scratch13L7U/ordinary only source recipe required')
 if not r['default_head_joint'] or r['target_access'] or r['U_truth_read'] or r['head_fit_updates'] or r['checkpoint_sources'] or r['As_training_observations'] or r['actual_steps']!=4440 or r['L_forward_packets']!=4546560 or r['extra_L_forward_packets']!=1136640 or r['supcon_applied']!=3330 or r['total_pair_weight']!=1.25:raise ValueError('Fixed nominal raw8L/1.25 control only')
 return dict(config=c,args=args,result=r,contract=contract)
