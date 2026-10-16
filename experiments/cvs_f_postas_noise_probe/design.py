"""Fixed post-As source-only recoverability probe;no training or new head."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261017-diagnostic-FcrossRX-postAs-noise-control-fitL-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_postAs_noise_control_fitL_20261017_r01'
OWNER='codex/root/FcrossRX-postAs-noise-control-20261017'
BASE=Path(PROJECT)/'runs'/RUN
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION=Path(PROJECT)/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
BACKEND=PROJECT+'/releases/cvs_FcrossRX_noisepair_sourceRX1_E20_20261017_r01'
BACKEND_COMMIT='ee305facac7874bfc1340291ee5360a3665621d9'
REFERENCE_RUN='20261017-phase1-FcrossRX-noisepair-sourceRX1-e20-s2026092701-m3-r01'
SEED=2026092701;FIT=(3,4,6,8);ARMS=('RAW10','ANTI');VIEWS=('clean','full','no_added_noise_fixed_receiver');BATCH=128
STREAMS=dict(order=20261017601,channel=20261017602,noise=20261017603,receiver=20261016102)
FULL_FP32_POLICY=dict(cudnn_allow_tf32=False,cuda_matmul_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=False,matmul_precision='highest')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def rows():return [dict(row_id=f'frozen-{a}-s{SEED}',arm=a,model_seed=SEED) for a in ARMS]
def reference(arm):
 if arm not in ARMS:raise ValueError('Only registered frozen own-default-head models')
 return Path(PROJECT)/'runs'/REFERENCE_RUN/f'{arm}-RX1-s{SEED}'/'source'
def config(row):
 if row not in rows():raise ValueError('Only two frozen registered models,one model seed')
 return dict(schema='FcrossRX_postAs_noise_control_fitL_v1',run_id=RUN,launch_owner=OWNER,**row,output_root=(BASE/row['row_id']).as_posix(),source_contract=SOURCE,source_preparation=PREPARATION.as_posix(),checkpoint_source=reference(row['arm']).as_posix(),backend=BACKEND,backend_commit=BACKEND_COMMIT,fit_RX=list(FIT),held_RX1_used=False,source_L_count=5040,source_TX=list(range(6)),source_days=[1,2,3],batch_size=BATCH,streams=dict(STREAMS),views=list(VIEWS),scene='practical_low_urban',channel='Unchanged original25MHz/IQ256/fc2462MHz/residual/post_sync/EQoffG1/reference30/reflect;one nominal lowurban draw peractualfitL',controlled_effect='Retain original full receiver state,compensation,H,residualkernel and clipped AGC scalar;remove only traced newly-added noise before that same scalar;no rerun/recomputed Q or AGC',clean='Original normalized source recording,unknown ground RX/TX effects intact',network='Frozen eval own defaultcos30/joint160/F+CROSS_RX/SIG_TIME_X3/unifiedSinc/NoExplicitCFO;no newhead or fittedparameters',observer='Call-local existing projection/gate hooks,original forward_identity_only/defaulthead;sum unsmoothed CE feature derivatives via autograd.grad,no .grad accumulation or optimizer',primary='ANTI',decision='ANTI BLOCKED and unlocked must EACH restore>=half of matched full->clean error gap underfixedreceiver no-added-noise control. Gap<=0 or missing stratum is unsupported. RAW10 descriptive only. Failure closes current default-head noise-masking rationale;not an information-theoretic bound or allpossible denoising impossibility.',source_only=True,target_access=False,U_used=False,V_used=False,cached_carrier_L6300_V_and_old_features_materialized=True,only_fitL5040_IQ_values_consumed=True,old_cached_features_consumed=False,optimizer_updates=0,head_fit_updates=0,EMA_updates=0,model_epochs=0,automatic_training=False,expected_NN=15122,expected_decisions=15120,expected_autograd_calls=120,expected_autograd_packet_equivalents=15120,expected_batches=40,formal_test_changed=False,stop_rule='Only own systemic role/provenance/nonfinite/frozen-state/budget/collision/unlaunchable failures;preservepartial,no auto retry or unrelated healthy-process action')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed diagnostic roles/views/streams/source/model budget differs')
 return c

def verify_reference(arm,done,init,contract,freeze,carrier,result,expected):
 if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access'] or done.get('target_evaluated') or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']:raise ValueError('Known scratch source-only own default head FINAL20 required')
 c=done['config']
 if c!=expected or c['arm']!=arm or c['run_id']!=REFERENCE_RUN or c['model_seed']!=SEED or c['output_root']!=reference(arm).as_posix() or c['checkpoint_sources'] or c['separate_head_training'] or c['explicit_CFO_identity_feature']:raise ValueError('Matching new10L reference source differs')
 if contract['held_RX']!=1 or contract['source_receivers']!=list(FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Exact fit4RX source roles required')
 roles=contract['role_ids']
 if any(len(v)!=len(set(v)) or not set(v)<=set(carrier['role_ids'][k]) for k,v in roles.items()) or set(roles['L_s'])&(set(roles['U_s'])|set(roles['V'])) or set(roles['U_s'])&set(roles['V']):raise ValueError('Source roles overlap or escape unchanged carrier')
 if result['actual_steps']!=4440 or result['L_forward_packets']!=5683200 or not result['default_head_joint'] or result['head_fit_updates'] or result['target_access'] or result['checkpoint_sources'] or result['U_truth_read']:raise ValueError('Complete own jointly trained default head required')
 return contract

def recoverability(labels,predictions,physics):
 n=len(labels)
 if n<1 or set(predictions)!=set(VIEWS) or any(len(v)!=n for v in predictions.values()) or len(physics)!=n:raise ValueError('Full matched three-view source decisions required')
 if any(type(y) is not int or y not in range(6) for y in labels) or any(type(y) is not int or y not in range(6) for v in predictions.values() for y in v):raise ValueError('All six registered classes required')
 cells={}
 for dimension,value in (('state','BLOCKED'),('lock','unlocked')):
  ix=[i for i,p in enumerate(physics) if p[dimension]==value];correct={v:sum(predictions[v][i]==labels[i] for i in ix) for v in VIEWS};gap=correct['clean']-correct['full'];recovered=correct[VIEWS[-1]]-correct['full'];supported=bool(ix) and gap>0 and 2*recovered>=gap
  cells[value]=dict(count=len(ix),correct=correct,full_to_clean_correct_gap=gap,recovered_correct=recovered,recovery_fraction=recovered/gap if gap>0 else None,supported=supported)
 return dict(primary='ANTI',strata=cells,supported=all(v['supported'] for v in cells.values()),not_information_theoretic_upper_bound=True,automatic_training=False)
