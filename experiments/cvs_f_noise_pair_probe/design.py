"""Noise-pair variance rationale only: frozen default heads, zero updates."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261017-diagnostic-FcrossRX-antithetic-noise-fitL-m2-r01'
RELEASE='cvs_FcrossRX_antithetic_noise_fitL_20261017_r01'
OWNER='codex/root/FcrossRX-antithetic-noise-20261017'
BASE=Path(PROJECT)/'runs'/RUN
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION=Path(PROJECT)/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
SEEDS=(2026092701,2026092702);FIT=(3,4,6,8);BATCH=128;DRAWS=8
STREAMS=dict(order=20261017401,channel=20261017402,noise=20261017403,receiver=20261016102)
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
FULL_FP32_POLICY=dict(cudnn_allow_tf32=False,cuda_matmul_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=False,matmul_precision='highest')
REFERENCES={2026092701:dict(run='20261016-phase1-F-crossRX-additiveAs-sourceRX1-e20-s2026092701-m3-r01',backend='cvs_F_crossRX_additiveAs_sourceRX1_e20_20261016_r01',commit='7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b'),2026092702:dict(run='20261016-phase1-F-crossRX-additiveAs-confirm-sourceRX1-e20-s2026092702-m3-r01',backend='cvs_F_crossRX_additiveAs_confirm_sourceRX1_e20_20261016_r01',commit='05984a1229ebac7025a95081dc5fae4108276b28')}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def rows():return [dict(row_id=f'frozen-D_raw-s{s}',model_seed=s,arm='noise_pair_variance') for s in SEEDS]
def reference(seed):return Path(PROJECT)/'runs'/REFERENCES[seed]['run']/f'D_raw-RX1-s{seed}'/'source'
def config(row):
 if row not in rows():raise ValueError('Only two frozen registered source seeds allowed')
 seed=row['model_seed']
 return dict(schema='FcrossRX_antithetic_noise_fitL_v1',run_id=RUN,launch_owner=OWNER,**row,output_root=(BASE/row['row_id']).as_posix(),source_contract=SOURCE,source_preparation=PREPARATION.as_posix(),checkpoint_source=reference(seed).as_posix(),checkpoint_role='Existing scratchE20 D_raw baseline,read-only diagnostic;no new student/teacher initialization',backend=REFERENCES[seed],fit_RX=list(FIT),held_RX1_used=False,source_L_count=5040,source_TX=list(range(6)),source_days=[1,2,3],batch_size=BATCH,draws=DRAWS,streams=STREAMS,scenes=list(SCENES),channel='Original residual/post_sync/EQoff G1/25MHz/fc2462MHz/reference30/reflect,uniform nominal six scenes;fixed per-packet all non-noise latents;RX/day/eq hardware session,not TX',noise_pair='For each of8 independent named primitive standard-normal draws use +z and -z before original RX/IQ/G processing;each view recomputes original AGC;no output averaging,attenuation,denoising or post-hoc sign flip',network='Fixed F+CROSS_RX,NoExplicitCFO,SIG_TIME_X3/unifiedSinc/joint160/defaultcos30;eval deterministic frozen parameters and buffers',loss='Per-batch mean native default-head CE(label_smoothing=.01);no RDrop KL,SupCon or dropout in this isolated CE derivative observer',gradient='All trainable parameter derivatives,no .grad accumulation;Sinc q/25000 physicalHz;whole sum of5disjoint groups',statistics='Unbiased within-fixed-batch trace estimates across8 noise draws: V_ind=(Vplus+Vminus)/4,V_ant=var((gplus+gminus)/2);covariance_identity V_ant=V_ind+C/2. Control estimated from same marginal derivative samples,no extra model forwards',decision='Both source model seeds must have summed whole V_ant/V_ind<=.80 and negative noise covariance in>=60% of40batches. Only licenses separate training preregistration;else abandon this variance rationale,no layer/state/RX/seed/draw/threshold rescue',source_only=True,target_access=False,U_used=False,V_used=False,old_source_cache_L6300_V_and_parent_features_materialized=True,only_fitL5040_values_consumed=True,optimizer_updates=0,head_fit_updates=0,EMA_updates=0,model_epochs=0,automatic_training=False,expected_NN=5040*2*DRAWS+2,expected_decisions=5040*2*DRAWS,expected_batches=40,stop_rule='Role/provenance/nonfinite/state-mutation/collision/unlaunchable failure preserves partials,no automatic retry;no poor-metric technical stop or unrelated process action')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed source/noise/seed/zero-update design differs')
 return c

def verify_reference(seed,done,init,contract,freeze,carrier):
 c=done['config'];r=reference(seed)
 if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access'] or done.get('target_evaluated') or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']:raise ValueError('Verified scratch source-only final20 provenance required')
 if c['model_seed']!=seed or c['row_id']!=f'D_raw-RX1-s{seed}' or c['run_id']!=REFERENCES[seed]['run'] or c['arm']!='D_raw' or c['epochs']!=20 or c['updates']!=4440 or c['head_fit_updates'] or c['target_access'] or c['checkpoint_sources'] or c['output_root']!=r.as_posix():raise ValueError('Wrong matching D_raw reference config')
 if contract['held_RX']!=1 or contract['source_receivers']!=list(FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Effective source fit contract differs')
 if any(len(v)!=len(set(v)) or not set(v)<=set(carrier['role_ids'][k]) for k,v in contract['role_ids'].items()) or set(contract['role_ids']['L_s'])&(set(contract['role_ids']['U_s'])|set(contract['role_ids']['V'])) or set(contract['role_ids']['U_s'])&set(contract['role_ids']['V']):raise ValueError('Physical source roles overlap/escape carrier')
 if c['training_mechanism']['As'] or c['training_mechanism']['Ag'] or c['training_mechanism']['CROSS_RX'] is not True or c['training_mechanism']['sinc_coordinates']!='CENTER_25K' or c['feature_extractor']['output_embedding']!=160 or c['explicit_CFO_identity_feature'] or c['separate_head_training']:raise ValueError('Fixed unaugmented F+CROSS_RX default head required')
 return contract
