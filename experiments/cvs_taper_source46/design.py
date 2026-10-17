"""One fixed native time-Tukey bypass source diagnostic, never training."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261018-diagnostic-FcrossRX-nativeTaper-L6300-s2026092701-m2-r01'
RELEASE='cvs_FcrossRX_nativeTaper_L6300_20261018_r01'
BASE=Path(PROJECT)/'runs'/RUN
REFERENCE=Path(PROJECT)/'runs/20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
SOURCE=Path(PROJECT)/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
CACHE=Path(PROJECT)/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701/source_cache.pt'
RX=(1,3,4,6,8);ARMS=('B00','B01');SEED=2026092701
DRAW_SEED=202610180461;ORDER_SEED=202610180462
OWNER='codex/root/native-taper-source46'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def config():
 return dict(schema='native_taper_source46_v1',run_id=RUN,owner=OWNER,arms=list(ARMS),model_seed=SEED,source_RX=list(RX),source_days=[1,2,3],TX=list(range(6)),L=6300,U_hidden=56700,V=27000,source_RX_holdout=None,source_only=True,role='L_s',U_used=False,V_used=False,target_access=False,source_contract=str(SOURCE),source_cache=str(CACHE),reference=str(REFERENCE),IQ_shape=[2,256],fs_Hz=25000000,ordinary_epoch=13,conditions=['RAW','FULL','NO_TAPER'],draws=2,draw_seed=DRAW_SEED,order_seed=ORDER_SEED,batch_size=128,intervention='Call original _tukey_window then return ones only for NO_TAPER;unchanged gate/RNG/othernativePA+DC+channel',new_IQ_operator=False,new_models=0,training_updates=0,teacher_NN=0,head_fit_updates=0,new_head=False,expected_NN=63004,automatic_training=False,criterion=dict(each_model_draw_retention_gain_min_pp=.2,each_model_draw_overall_accuracy_delta_min_pp=0,each_model_positive_RX_min=4,each_model_worst_RX_gain_min_pp=-.25),source_claim='Frozen input effect only;no Ag or As training gain proved')
def validate(c):
 if c!=config():raise ValueError('Fixed allsource/L-only/frozen/noquery contract differs')
 return c
