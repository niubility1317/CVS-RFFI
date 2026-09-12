"""Read-only N607 process, GPU and exact input metadata inspection."""
import json
import os
from pathlib import Path
import subprocess
project=Path('/home/szu2070436088/2510044040/CV-SincNet')
result={'python':str(Path('/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python').resolve()),'processes':[]}
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        cmd=(p/'cmdline').read_bytes().decode().split('\0')
        if any('train' in v or 'dispatch' in v for v in cmd) and any('python' in v for v in cmd[:2]):
            env=dict(s.split('=',1) for s in (p/'environ').read_bytes().decode().split('\0') if '=' in s)
            result['processes'].append({'pid':int(p.name),'cwd':os.readlink(p/'cwd'),'entry':cmd[:3],
                'gpu':env.get('CUDA_VISIBLE_DEVICES'),'out':cmd[cmd.index('--output_dir')+1] if '--output_dir' in cmd else None})
    except (PermissionError,FileNotFoundError,ProcessLookupError,UnicodeError):pass
for name,query in [('gpus','index,uuid,memory.used,memory.total,utilization.gpu'),('compute','gpu_uuid,pid,used_memory')]:
    opt='--query-gpu=' if name=='gpus' else '--query-compute-apps='
    result[name]=subprocess.check_output(['nvidia-smi',opt+query,'--format=csv,noheader,nounits'],text=True).splitlines()
for rel in ['Dataset_WigSig/ManySig.pkl','runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/target_inputs/manifest.json',
            'runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/target_inputs/iq.npy',
            'runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/target_truth/truth_sidecar.json']:
    p=project/rel;result[rel]={'exists':p.is_file(),'bytes':p.stat().st_size if p.is_file() else None}
    if p.name=='manifest.json' and p.is_file():
        data=json.loads(p.read_text());result[rel]['metadata']={k:v for k,v in data.items() if k!='sample_ids'}
print(json.dumps(result,indent=2))
