"""Read only existing model-bound aggregate metadata, never source samples."""
import json
from pathlib import Path
import subprocess

FLAGS = ['-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
REMOTE = r'''
import json
from pathlib import Path
import numpy as np
base=Path('/home/szu2070436088/2510044040/CV-SincNet')
rows=[]
for seed in range(2026092701,2026092705):
 name=f'cvs-daot-rc4-s{seed}'
 ground=base/'runs/20260927-phase2-cvs-d92-practical-manytx-m5-r01'/name/'ground'
 source=base/'runs/20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01'/name
 manifest=json.loads((ground/'manifest.json').read_text())
 npz=ground/'int8_domain_class_center_lowrank_residual_radius_v2.npz'
 with np.load(npz,allow_pickle=False) as d:
  members={k:dict(shape=list(d[k].shape),dtype=str(d[k].dtype),bytes=d[k].nbytes) for k in d.files}
  classes=d['class_registry'].astype(str).tolist()
  domains=d['domain_registry'].astype(str).tolist()
 rows.append(dict(seed=seed,ground=str(ground),manifest=manifest,members=members,
  classes=classes,domains=domains,npz_file_bytes=npz.stat().st_size,
  manifest_file_bytes=(ground/'manifest.json').stat().st_size,
  checkpoint_file_bytes=(source/'final_ssdg.pth').stat().st_size,
  initialization=json.loads((source/'initialization.json').read_text()),
  completion=json.loads((source/'completion.json').read_text())))
print(json.dumps(dict(rows=rows,source_samples_read=False,source_features_read=False,target_read=False)))
'''

def main():
    command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -'
    result=subprocess.run(['ssh',*FLAGS,'-T','N607',command],input=REMOTE.encode(),capture_output=True,check=True)
    data=json.loads(result.stdout)
    root=Path('E:/type10-7/local_artifacts/d92_upgrade_20260928')
    path=root/'ground_payload_readback.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(data,stream,ensure_ascii=False,indent=2)
    for row in data['rows']:
        numeric=sum(v['bytes'] for v in row['members'].values() if v['dtype'].startswith(('int','float')))
        print(json.dumps(dict(seed=row['seed'],classes=row['classes'],domains=len(row['domains']),
            numeric_bytes=numeric,npz_file_bytes=row['npz_file_bytes'],
            manifest_file_bytes=row['manifest_file_bytes'],checkpoint_file_bytes=row['checkpoint_file_bytes'],
            checkpoint_sha256=row['manifest']['checkpoint_sha256'],
            provenance=row['manifest']['provenance_status'])))
    print(path)

if __name__=='__main__':main()
