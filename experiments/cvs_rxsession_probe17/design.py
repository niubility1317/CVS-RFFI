"""One fixed frozen default-head source hypothesis;no training or target."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261017-diagnostic-FcrossRX-RXsession-fitL-s2026092701-m1-r01'
RELEASE='cvs_FcrossRX_RXsession_fitL_20261017_r01'
OWNER='codex/root/FcrossRX-RXsession-source-20261017'
BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_FcrossRX_AsTiming_sourceRX1_E20_20261017_r01'
BACKEND_COMMIT='f788309f5f9e27689ec1e98ae99a6bd83237e1f9'
REFERENCE_RUN='20261017-phase1-FcrossRX-AsTiming-sourceRX1-e20-s2026092701-m4-r01'
REFERENCE=PROJECT+'/runs/'+REFERENCE_RUN+'/ALL-RX1-s2026092701/source'
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION=PROJECT+'/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
MODEL_SEED=2026092701;CHANNEL_SEED=202610171911;RECEIVER_SEED=20261016102
FIT=(3,4,6,8);DAYS=(1,2,3);TX=tuple(range(6));SCENES=('practical_mid_urban','practical_low_urban');BANKS=('original','bank1','bank2','bank3');BATCH=128
POLICY=dict(cudnn_allow_tf32=False,cuda_matmul_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=False,matmul_precision='highest')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def rows():return [dict(row_id='frozen-ALL-s2026092701',arm='ALL',model_seed=MODEL_SEED)]
def config(row):
 if row not in rows():raise ValueError('One fixed own As-trained source checkpoint only')
 return dict(schema='F_CROSS_RX_session_sensitivity_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']).as_posix(),backend=BACKEND,backend_commit=BACKEND_COMMIT,checkpoint_source=REFERENCE,source_contract=SOURCE,source_preparation=PREPARATION,source_only=True,target_access=False,fit_RX=list(FIT),source_days=list(DAYS),source_TX=list(TX),L=5040,U=0,V=0,held_RX1=0,scenes=list(SCENES),banks=list(BANKS),streams=dict(channel=CHANNEL_SEED,receiver=RECEIVER_SEED),batch_size=BATCH,network='FrozenF+CROSS_RX/SIG_TIME_X3/unifiedSinc/joint160/defaultcos30/NoExplicitCFO;original model eval;no newhead',channel='Original full practical residual/post_sync/fs25MHz/fc2462MHz/IQ256/EQoffG1/reference30/reflect;one H/underlying noise draw perpacket/scene,4RX parameter realizations plus exact original replay',session='physical RX/day/eq only,bank suffix1/2/3;stable acrossallTX/packets/scenes,no TX/prediction/state defined session',controls='Identical H/noise RNG/quality/status/scale and original replay;processed noise may vary through same RX IQ operator,AGC recomputed faithfully',criterion=dict(primary='As-trained ALL',per_scene_nominal_accuracy_advantage_over_3bank_mean_min_pp=1.,per_scene_mean_nominal_to_bank_hard_disagreement_min=.03,each_bank_gain_over_nominal_max_pp=.5,all_registered_scenes_required=True),on_fail='Close fixedsession-undercoverage rationale;no different banks/seed/scenes/RX/threshold rescue,no automatic training',on_pass='Only source-fit session sensitivity rationale;separately register matching session coverage training,not Ag/target/held/crossseed or identitygain',model_epochs=0,optimizer_updates=0,head_fit_updates=0,teacher_forwards=0,EMA_updates=0,expected_NN=45362,expected_decisions=45360,expected_physics_records=10080,expected_native_process_calls=50400,expected_receiver_sessions=48,automatic_training=False,formal_test_changed=False,stop_rule='Only own role/provenance/nonfinite/wrongbudget/frozenstate/outputcollision/unlaunchable failures;preserve outputs,no automatic retry/no unrelated process action')
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Fixed checkpoint/data/view/seed/budget/criterion only')
 return c

def verify_reference(done,init,freeze,contract,carrier,result,expected):
 if done['config']!=expected or expected['run_id']!=REFERENCE_RUN or expected['arm']!='ALL' or expected['model_seed']!=MODEL_SEED:raise ValueError('Exact original ALL runtime configuration required')
 if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access'] or done['target_evaluated'] or not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or init['source_roles']!='EXACT_MATCH':raise ValueError('Known scratch source-only FINAL20 provenance only')
 if freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access'] or freeze['checkpoint']!=REFERENCE+'/final_ssdg.pth':raise ValueError('Exact final-only default head checkpoint required')
 if contract['held_RX']!=1 or contract['source_receivers']!=list(FIT) or contract['native_role_comparison']!='EXACT_MATCH_CARRIER_THEN_SOURCE_FOLD' or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Matching fit4RX effective LUV required')
 rr=contract['role_ids']
 if any(len(v)!=len(set(v)) or not set(v)<=set(carrier['role_ids'][k]) for k,v in rr.items()) or set(rr['L_s'])&(set(rr['U_s'])|set(rr['V'])) or set(rr['U_s'])&set(rr['V']):raise ValueError('Exact legal source role ID subsets required')
 if result['actual_steps']!=4440 or result['L_forward_packets']!=4546560 or result['As_training_observations']!=568320 or not result['default_head_joint'] or result['head_fit_updates'] or result['target_access'] or result['U_truth_read'] or result['checkpoint_sources']:raise ValueError('Unmodified jointly trained ALL defaulthead source required')
 return contract

def criteria(groups):
 if set(groups)!=set(SCENES):raise ValueError('Both complete scenes required')
 checks={}
 for scene,z in groups.items():
  if z['count']!=5040 or set(z['banks'])!=set(BANKS):raise ValueError('Complete all-packet four-bank decisions required')
  ref=z['banks']['original'];other=[z['banks'][b] for b in BANKS[1:]]
  checks[scene+'/advantage']=100*(ref['accuracy']-sum(q['accuracy'] for q in other)/3)>=1.-1e-10
  checks[scene+'/disagreement']=sum(q['disagreement_vs_original'] for q in other)/3>=.03-1e-10
  checks[scene+'/no_better_bank']=all(100*(q['accuracy']-ref['accuracy'])<=.5+1e-10 for q in other)
 return checks
