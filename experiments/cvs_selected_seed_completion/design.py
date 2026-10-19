"""User-selected historical configurations; change only model seed and run binding."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def runtime():
 p=ROOT/'runtime.json'
 if not p.is_file():p=ROOT/'local_artifacts/selected_seed_runtime.json'
 return json.loads(p.read_text(encoding='utf8'))
PROJECT=Path(runtime()['project'])
RUN='20261019-phase1-F-selected-eight-e30-s0203-m16-r01'
RELEASE='cvs_F_selected_eight_E30_s0203_20261019_r01'
OWNER='codex/root/selected-eight-three-seeds'
BASE=PROJECT/'runs'/RUN
SEEDS=(2026092702,2026092703)
GPU_IDS=(4,6)
MAX_GPU_WORKERS=3
VIEWS=('clean',)
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
CLASSES=6
FORMAL_VIEWS=PROJECT/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views'
TRUTH=PROJECT/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
SOURCE=PROJECT/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
FULL_FP32_POLICY=dict(cudnn_allow_tf32=False,cuda_matmul_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=False,matmul_precision='highest')
GROUPS=(
 ('PAOFF','pa','AS_PAOFF','20261018-phase1-FcrossRX-PAwaveform-tail-e30-s2026092701-m2-r01'),
 ('HALF_AS','clock','B01','20261017-phase1-F-crossRX-clock-e30-score-recovery-s2026092701-m5-r01'),
 ('B0','u','B0','20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01'),
 ('URAW','u','URAW','20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01'),
 ('HARD_LCE','hard','H01','20261018-phase1-FcrossRX-hardLCE-e30-s2026092701-m2-r01'),
 ('FIR_AS','fir','B11','20261017-phase1-F-crossRX-spectralIQ-firsttrain-e20plus10-s2026092701-m2-r01'),
 ('ORD128','cross','ORD128','20261018-phase1-FcrossRX-crossviewSupCon-e30-s2026092701-m3-r01'),
 ('DUP192','cross','DUP192','20261018-phase1-FcrossRX-crossviewSupCon-e30-s2026092701-m3-r01'))
def rows():
 out=[]
 for seed in SEEDS:
  for group,family,arm,old_run in GROUPS:
   prefix='FIR_'+arm if family=='fir' else arm
   out.append(dict(group=group,family=family,arm=arm,model_seed=seed,row_id=prefix+'-s'+str(seed),seed01_run=old_run,seed01_row=prefix+'-s2026092701'))
 return out
def row(identifier):
 found=next((r for r in rows() if r['row_id']==identifier),None)
 if found is None:raise ValueError('Only requested eight configurations and missing seeds02/03 are authorized')
 return found
def row_ok(value):
 if value not in rows():raise ValueError('Unregistered replication row')
 return value
def dependencies(r):
 row_ok(r)
 return ['B01-s'+str(r['model_seed'])] if r['family']=='fir' else []
def config(r,segment=30):
 row_ok(r)
 from .binding import configured
 b=configured(r)
 return b['config'](segment)
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(path)
