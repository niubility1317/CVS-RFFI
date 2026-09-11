"""Read-only direct-N607 input/runtime/process inspection."""
import subprocess
PAYLOAD=r'''
import json,os,subprocess
from pathlib import Path
import torch,numpy
root=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/core90_evidence_frozen_392005_20260911')
payload=torch.load(root/'H0/ground_checkpoint.pt',map_location='cpu',weights_only=True)
contract=json.loads((root/'data_contract.json').read_text())
try:
    torch.ones(1).numpy();interop='PASS'
except Exception as exc:interop=type(exc).__name__+': '+str(exc)
processes=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():continue
    try:
        argv=(entry/'cmdline').read_bytes().split(b'\0');argv=[x.decode(errors='replace') for x in argv if x]
        if not argv or not any('run_core90_anchored' in x for x in argv):continue
        processes.append(dict(pid=int(entry.name),cwd=str((entry/'cwd').resolve()),argv=argv))
    except (OSError,PermissionError):pass
print(json.dumps(dict(torch=torch.__version__,numpy=numpy.__version__,numpy_interop=interop,
    lineage=payload['checkpoint_lineage'],contract_equal=payload['training_data_contract']==contract,
    dataset_id=contract['dataset_id'],counts={k:len(v) for k,v in contract['roles'].items()},
    source_receivers=contract['source_receivers'],source_days=contract['source_days'],
    split_seed=contract['split_seed'],equalized=contract['equalized'],tx_mapping=contract['tx_mapping'],
    architecture=payload['baseline_args'],anchored_processes=processes),indent=2))
'''
if __name__=='__main__':
    result=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
        'N607','/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input=PAYLOAD.encode('utf-8'),capture_output=True,timeout=30)
    print(result.stdout.decode('utf-8'));print(result.stderr.decode('utf-8',errors='replace'))
    raise SystemExit(result.returncode)
