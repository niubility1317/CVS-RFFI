"""Read-only ManyTx metadata coverage; no checkpoint, IQ transform or scoring."""
import json
from pathlib import Path
import subprocess

REMOTE=r'''
import json,sys
from pathlib import Path
base=Path('/home/szu2070436088/2510044040/CV-SincNet')
release=base/'releases/phase2_practical_data_20260927_r01'
sys.path.insert(0,str(release/'code'))
from dataset_wisig import load_wisig_compact_pkl
cfg=json.loads((release/'configs/phase2_practical_data_20260927.json').read_text())
raw=load_wisig_compact_pkl(cfg['manytx_path'])
txs=list(map(str,raw['tx_list']));rxs=list(map(str,raw['rx_list']));eq=list(raw['equalized_list']).index(1)
rows=[]
for ri,rx in enumerate(rxs):
 if rx in cfg['source_receivers'] or rx in cfg['target_receivers']:continue
 counts=[]
 for tx in cfg['old_classes']+cfg['new_classes']:
  ti=txs.index(tx)
  counts.append(sum(len(v[eq]) for v in raw['data'][ti][ri] if v[eq] is not None))
 rows.append(dict(receiver=rx,min_records=min(counts),classes_with_at_least198=sum(n>=198 for n in counts),class_count=len(counts)))
print(json.dumps(dict(scope='not used by matched20260927 source/target receiver sets; historical all-time exposure NOT certified',
 checkpoint_read=False,truth_read=False,model_scores_read=False,unused_receiver_coverage=rows)))
'''


def main():
    compile(REMOTE,'remote','exec')
    p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607','/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -'],input=REMOTE.encode('utf-8'),capture_output=True)
    if p.returncode:raise RuntimeError(p.stderr.decode('utf-8'))
    data=json.loads(p.stdout)
    path=Path('E:/type10-7/local_artifacts/d92_upgrade_20260928/confirmation_inventory.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(data,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
