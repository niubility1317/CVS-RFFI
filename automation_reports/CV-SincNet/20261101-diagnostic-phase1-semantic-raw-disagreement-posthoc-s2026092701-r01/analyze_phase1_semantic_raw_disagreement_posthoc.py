"""Post-hoc diagnostic; never a deployable or blind threshold selection."""
import subprocess
from pathlib import Path
remote=r'''
from pathlib import Path
import json,numpy as np
r=Path('/home/szu2070436088/2510044040/CV-SincNet')
new=r/'runs/20261101-phase1-semantic-consensus-s2026092701-r03/flow/test_prediction'
dura=r/'runs/20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction'
for p in (new,dura):
 complete=json.loads((p/'complete.json').read_text());assert complete['status']=='PREDICTIONS_COMPLETE' and complete['TX_truth_reads']==0
ids=np.load(new/'physical_ids.npy',allow_pickle=False);assert len(ids)==168000 and np.array_equal(ids,np.load(dura/'physical_ids.npy',allow_pickle=False))
views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban']
pred={}
for v in views:
 raw=np.load(new/(v+'.raw.npy'),allow_pickle=False);base=np.load(dura/(v+'.restored.npy'),allow_pickle=False);sem=np.load(new/(v+'.restored.npy'),allow_pickle=False)
 assert all(a.shape==(len(ids),) for a in (raw,base,sem));pred[v]=(raw,base,sem)
print('ALL_FIXED_PREDICTIONS_READ_BEFORE_TRUTH',flush=True)
truth=json.loads((r/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json').read_text())
y=np.asarray([truth[str(i)]['label'] for i in ids]);result={}
for view,(raw,base,sem) in pred.items():
 rows=[]
 for k in range(6):
  mask=(raw==k)&(raw!=sem);good=(raw==y);old=(sem==y)
  rows.append(dict(raw_pred=k,n=int(mask.sum()),raw_correct=int((mask&good).sum()),semantic_correct=int((mask&old).sum()),repairs=int((mask&good&~old).sum()),damages=int((mask&~good&old).sum())))
 result[view]=rows
print('PHASE1_POSTHOC_RAW_DISAGREEMENT '+json.dumps(dict(status='POSTHOC_DEVELOPMENT_TEST_DIAGNOSTIC',views=result,caveat='Observed test labels; no gate or threshold is validated and all subsequent uses of this test are nonblind.'),ensure_ascii=False))
'''
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=120)
stdout=p.stdout.decode('utf8',errors='replace');print(stdout);print(p.stderr.decode('utf8',errors='replace'))
if p.returncode==0:
 line=next(x[len('PHASE1_POSTHOC_RAW_DISAGREEMENT '):] for x in stdout.splitlines() if x.startswith('PHASE1_POSTHOC_RAW_DISAGREEMENT '))
 path=Path(__file__).parent/'phase1_semantic_raw_disagreement_posthoc.json'
 if path.exists():
  import json
  assert json.loads(path.read_text(encoding='utf8'))==json.loads(line)
 else:path.write_text(line+'\n',encoding='utf8',newline='\n')
raise SystemExit(p.returncode)
