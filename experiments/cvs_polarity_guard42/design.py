"""One fixed source-only necessary identity guard, not an augmentation gain claim."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN='20261018-diagnostic-FcrossRX-polarity-sourceV-s2026092701-m1-r01'
RELEASE='cvs_FcrossRX_polarity_guard_sourceV_20261018_r01'
OWNER='codex/root/polarity-source-guard42'
BASE=PROJECT/'runs'/RUN
REFERENCE=PROJECT/'releases/cvs_FcrossRX_crossviewSupCon_E30_20261018_r01'
REFERENCE_COMMIT='ec4c5d7f46aedb5aaece77fdd72dbe6ed4c157f2'
REFERENCE_ROW='ORD128-s2026092701'
SMOKE_SEED=202610185201
N=1800
RX=(1,3,4,6,8)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')
def config():
 return dict(schema='P42_polarity_source_guard_v1',run_id=RUN,launch_owner=OWNER,source_only=True,scope='Necessary frozen default-head identity guard only;not Ag/generalization or training evidence',checkpoint_row=REFERENCE_ROW,checkpoint_rule='Prefixed ORD control from P41,own scratch20+own10/fixedfinal;not target-ranked',reference_code_commit=REFERENCE_COMMIT,source_RX=list(RX),source_days=[1,2,3],source_TX=list(range(6)),source_role='V',source_packets=N,full_source_contract='Original L6300/U56700hidden/V27000;all5 sourceRX;noLdev/heldoutRX',operator='z->conj(z);I unchanged,Q negated;wholepacket;no time/noise/filter/RMS/padding change',views=['raw','conjugate'],inverse_control='conjugate twice exactly raw;verified on all actual input tensors,0extraNN',guard=dict(max_global_accuracy_drop=.005,max_each_RX_accuracy_drop=.015,min_fixed_raw_correct_retention=.99),model_seed=2026092701,split_seed=392005,synthetic_smoke_seed=SMOKE_SEED,augmentation_seed=None,evaluation_seed=None,expected_NN=3602,real_source_NN=3600,synthetic_NN=2,optimizer_updates=0,head_fit_updates=0,teacher_NN=0,As_calls=0,U_TX_truth_read=False,target_access=False,automatic_training=False,automatic_seed_extension=False,goal_complete=False)
