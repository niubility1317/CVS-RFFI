"""One authorized native-Windows -> direct-N607 dispatch; no persistent SSH."""
import json
import subprocess

CONFIG='E:/type10-7/tools/n607_ssh_config'
REMOTE_PY='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
PAYLOAD='''import json, os, subprocess
from pathlib import Path
base=Path('/home/szu2070436088/2510044040/CV-SincNet')
release=base/'releases/core90_evidence_392005_edca15a6'
root=base/'runs/core90_evidence_frozen_392005_20260911'
log=base/'runs/core90_evidence_frozen_392005_20260911.coordinator.log'
pidfile=base/'runs/core90_evidence_frozen_392005_20260911.coordinator.pid'
if root.exists() or log.exists() or pidfile.exists():
    raise FileExistsError('run already landed or dispatch evidence exists; reconcile read-only')
script=release/'code/scripts/run_core90_evidence_experiment.py'
if not script.is_file():raise FileNotFoundError(script)
command=['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u',str(script),
    '--root',str(root),'--dataset',str(base/'Dataset_WigSig/ManySig.pkl'),
    '--seed','392005','--train-gpu','2','--head-gpus','0,1,2,3,4']
environment=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
with log.open('x',encoding='utf-8') as output:
    process=subprocess.Popen(command,cwd=release,env=environment,stdin=subprocess.DEVNULL,
        stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
with pidfile.open('x',encoding='ascii') as output:output.write(str(process.pid)+'\\n')
print(json.dumps({'pid':process.pid,'root':str(root),'cwd':str(release),'argv':command,'log':str(log)}))
'''


if __name__=='__main__':
    compile(PAYLOAD,'<reviewed_remote_dispatch>','exec')
    result=subprocess.run(['ssh','-F',CONFIG,'-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
                           'N607',REMOTE_PY,'-'],input=PAYLOAD.encode('utf-8'),capture_output=True,timeout=30)
    print(result.stdout.decode('utf-8',errors='strict'))
    if result.stderr:print(result.stderr.decode('utf-8',errors='replace'))
    raise SystemExit(result.returncode)
