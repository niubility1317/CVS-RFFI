"""Compare frozen choices with separately produced Phase1 expert decisions, without truth."""
import subprocess

remote=r'''
from pathlib import Path
import json,numpy as np
root=Path('/home/szu2070436088/2510044040/CV-SincNet/runs')
run='20261101-phase1-semantic-consensus-s2026092701-r03'
dest=root/run/'flow/test_prediction'
if not (dest/'complete.json').exists():print('WAITING_FOR_ALL_GUARDED_PREDICTIONS');raise SystemExit(0)
done=json.loads((dest/'complete.json').read_text());assert done['status']=='PREDICTIONS_COMPLETE' and done['TX_truth_reads']==0
experts=[None,'20261101-phase1-dura-trisigtime-baseline-s2026092701-r01',
         '20261101-phase1-boundary-student-E30-s2026092701-r02',
         '20261101-phase1-feature-conditioned-E30-s2026092701-r02']
ids=np.load(dest/'physical_ids.npy',allow_pickle=False);assert len(ids)==168000
others=[]
for name in experts[1:]:
 folder=root/name/'flow/test_prediction';other=json.loads((folder/'complete.json').read_text())
 assert other['status']=='PREDICTIONS_COMPLETE' and np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids)
 others.append(folder)
for view in done['views']:
 choices=np.load(dest/(view+'.choice.npy'),allow_pickle=False);out=np.load(dest/(view+'.restored.npy'),allow_pickle=False)
 raw=np.load(dest/(view+'.raw.npy'),allow_pickle=False);assert choices.shape==out.shape==raw.shape==(len(ids),)
 assert choices.dtype==np.uint8 and np.all(choices<4)
 predictions=[raw]+[np.load(folder/(view+'.restored.npy'),allow_pickle=False) for folder in others]
 expected=np.choose(choices,predictions);diff=int(np.sum(expected!=out))
 print('EXPERT_DECISIONS',view,'choices',[int(np.sum(choices==i)) for i in range(4)],'mismatches',diff,flush=True)
 assert diff==0,(view,diff)
print('VERIFIED_FIXED_EXPERT_DECISIONS',len(ids),'views',len(done['views']),'truth_reads',0,flush=True)
'''
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=120)
print(p.stdout.decode('utf8',errors='replace'));print(p.stderr.decode('utf8',errors='replace'));raise SystemExit(p.returncode)
