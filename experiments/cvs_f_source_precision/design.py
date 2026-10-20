"""Fixed source-cache-only covariance diagnostic; no target or model state."""
from pathlib import Path
import json
RUN='20261020-diagnostic-FcrossRX-source-withinTX-precision-m3-local-r01'
SOURCE_RUN='20261019-diagnostic-FcrossRX-feature-geometry-sourceV-m3-local-r01'
SOURCE_COMMIT='31467086c3ced30237f3ac86e91eae68645425d7'
SEEDS=(2026092701,2026092702,2026092703)
CLASSES=('14-10','14-7','20-15','20-19','6-15','8-20')
RX=(1,3,4,6,8);DAYS=(1,2,3);MODES=('RAW','TX','TX_RX_DAY')
PARTITION_SEED=2026102009
SHRINKAGE=.1;PRECISION_MIN=.25;PRECISION_MAX=4.
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def decision(rows):
 checks=[]
 for row in rows:
  pair=next(v for v in row['pairs'] if v['indices']==[1,3]);raw=pair['metrics']['RAW'];cur=pair['metrics']['TX']
  valid=raw['SNR']>0 and cur['cross_RX_signal']>0 and cur['SNR']>=1.1*raw['SNR'] and all(r is not None and r>0 for r in raw['per_TX_RX_spread_to_signal']) and all(c<=.95*r for c,r in zip(cur['per_TX_RX_spread_to_signal'],raw['per_TX_RX_spread_to_signal']))
  checks.append(dict(seed=row['model_seed'],PASS=bool(valid),SNR_ratio=cur['SNR']/raw['SNR'] if raw['SNR']>0 else None,per_TX_RX_spread_ratio=[c/r if c is not None and r is not None and r>0 else None for c,r in zip(cur['per_TX_RX_spread_to_signal'],raw['per_TX_RX_spread_to_signal'])]))
 return dict(status='SOURCE_DIAGNOSTIC_DECISION',primary='TX precision vs RAW for source-fixed weak pair1/3; TX_RX_DAY explanatory only',PASS=len(rows)==3 and all(v['PASS'] for v in checks),rows=checks,target_access=False,rule='All3 existing BASE model seeds: crossRX class-difference signal positive, SNR>=1.10xRAW and each weakTX RXspread/signal<=.95xRAW. This only motivates a separately preregistered L-only feature-loss study, never classifier substitution or target benefit.')
