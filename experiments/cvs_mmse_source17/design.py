"""One fixed full-source identity/observation guard;no target or training."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-diagnostic-F-MMSE-receiver-fullL-s2026092701-m1-r01'
RELEASE='cvs_F_MMSE_receiver_source_20261017_r01'
LOCAL='local_artifacts/F_MMSE_receiver_20261017'
OWNER='codex/root/F-MMSE-receiver-source'
BASE='PROJECT/runs/'+RUN
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
REFERENCE_RUN='20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02'
REFERENCE_ROW='CONCAT-s2026092701'
FIT=(1,3,4,6,8)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(alias):
 if not alias.startswith('PROJECT/'):raise ValueError('Registered project alias required')
 return Path(read(ROOT/'runtime.json')['project'])/alias[len('PROJECT/'):]
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
PROBE_SEED=20261017931
RECEIVER_SEED=20261016423
VIEWS=('raw',)+tuple(s+'/'+k for s in SCENES for k in ('noEQ','MMSE24','MMSE64'))
def config():
 return dict(run_id=RUN,launch_owner=OWNER,source_contract=SOURCE,preparation=PREPARATION,model_source=REFERENCE_RUN+'/'+REFERENCE_ROW+'/source',source_RX=list(FIT),source_RX_holdout=None,L=6300,U_TX_hidden=True,U_used=False,V_statistically_used=False,target_access=False,model_seed=2026092701,checkpoint_rule='Fixed ordinary fullAs source-final20;all6300/56700/27000;no target-based choice',question='Does existing coupled MMSE reception at matched24-sample target latency recover source identities versus nominal noEQ?',probe_seed=PROBE_SEED,receiver_seed=RECEIVER_SEED,scenes=list(SCENES),views=list(VIEWS),sample_rate_Hz=25000000.,carrier_Hz=2462000000.,reference_snr_db=30.,route='residual/post_sync',equalizer=dict(method='mmse',length=129,target_delay_candidate=24,target_delay_control=64,max_gain_db=20.,channel_estimation_nmse_db=-25.,estimation_model='Existing truth-plus-error engineering proxy,not a pilot estimator'),shared='Same H,geometry,state,shadow,scatter,receiver,sync,phase,base noise,quality,lock;only G and its induced signal/noise/AGC differ',quality_policy='Original H/input-power proxy;EQ skips unlocked exactly;never force lock',boundary='Originalreflect;same IQ256/no crop/shift/renormalize',expected_NN=63002,source_decisions=63000,new_model_epochs=0,training_updates=0,teacher_forwards=0,head_fit_updates=0,criteria=dict(min_low_gain_pp=.5,min_mid_gain_pp=.5,min_suburban_gain_pp=-.25,min_each_RX_low_gain_pp=-1.,min_low_original_errors_affected_share=.20),on_fail='Close fixed MMSE24 candidate;no64replacement/SNR/strength/latency/seed/threshold rescue',on_pass='Source receiver-view evidence only;no Ag/As training gain or joint result;matched fullsource cumulativeE30 training independently required',formal_test_changed=False,automatic_training=False,device='CPU',threads=2)
