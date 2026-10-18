"""One fixed seed, two own checkpoints, four frozen readouts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-diagnostic-FcrossRX-ownEMA-source-e50-s2026092701-m4-r02'
RELEASE='cvs_FcrossRX_ownEMA_source_E50_20261018_r02'
PARENT_RUN='20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01'
PARENT_RELEASE='cvs_FcrossRX_As_budget_E50_20261018_r01'
PARENT_COMMIT='0f399c910324e21a8b1a60346d879be5c2964a9c'
OWNER='codex/root/practical-ownEMA-source'
LOCAL='local_artifacts/F_ownEMA_source_E50_20261018_r02'
SEED=2026092701
RX=(1,3,4,6,8)
DAY=(1,2,3)
ARMS=('B00','B01')
LABELS=('NOAS_STUDENT','NOAS_EMA','AS_STUDENT','AS_EMA')
STATES=('model','ema_model')
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
HEAD='id_backbone.encoder.core.id_backbone.cls_head.weight'
EXPECTED_NN=133204

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def runtime():
 r=read(ROOT/'runtime.json')
 if r.get('run_id')!=RUN or r.get('parent_commit')!=PARENT_COMMIT:raise ValueError('Only fixed ownEMA source run and original E50 release')
 project=Path(r['project']);parent=Path(r['parent_release'])
 if not project.is_absolute() or parent!=project/'releases'/PARENT_RELEASE:raise ValueError('Absolute fixed project/parent release required')
 return r

def output(rt):return Path(rt['project'])/'runs'/RUN
def parent_source(rt,arm):
 if arm not in ARMS:raise ValueError('Only original B00/B01 seed01')
 return Path(rt['project'])/'runs'/PARENT_RUN/(arm+'-s'+str(SEED))/'source'
def panel_predictions(rt,arm):return parent_source(rt,arm).parent/'source_qualification'/'predictions.npz'

def assert_source_path(path,allowed):
 p=Path(path).resolve();roots=[Path(v).resolve() for v in allowed]
 if not any(p==v or p.is_relative_to(v) for v in roots):raise ValueError('Outside fixed source input/output scope')
 if any(part in {'target','targets','prediction','scoring','scores','formal','truth'} or part.startswith('analysis_auto_') for part in p.parts):raise ValueError('Formal/target paths forbidden in source-only readout')
 return p
