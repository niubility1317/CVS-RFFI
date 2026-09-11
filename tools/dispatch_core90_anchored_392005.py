"""Authorized one-shot N607 dispatch; a timeout must be reconciled, not retried."""
import argparse,json,subprocess


def payload(release):
    return 'release_path='+repr(release)+'\n'+r'''
import json,os,subprocess
from pathlib import Path
base=Path('/home/szu2070436088/2510044040/CV-SincNet')
release=Path(release_path).resolve()
if release.parent!=base/'releases':raise ValueError('release must be directly under project releases')
name='core90_anchored_s392005_20260911_r1'
root=base/'runs'/name
log=base/'logs'/(name+'.coordinator.log')
pidfile=base/'runs'/(name+'.coordinator.pid')
logs=base/'logs'/name
if root.exists() or log.exists() or pidfile.exists() or logs.exists():
    raise FileExistsError('run or dispatch artifacts already exist; reconcile read-only')
script=release/'code/scripts/run_core90_anchored_experiment.py'
if not script.is_file():raise FileNotFoundError(script)
old=base/'runs/core90_evidence_frozen_392005_20260911'
command=['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u',str(script),
    '--root',str(root),'--logs',str(logs),'--ground',str(old/'H0/ground_checkpoint.pt'),
    '--contract',str(old/'data_contract.json'),'--previous-source',str(old/'source_training.pt'),
    '--dataset',str(base/'Dataset_WigSig/ManySig.pkl'),'--seed','392005','--gpus','0,1,2,3,5','--threads','4']
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
with pidfile.open('x',encoding='ascii') as pidout:
    with log.open('x',encoding='utf-8') as output:
        process=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    pidout.write(str(process.pid)+'\n')
print(json.dumps(dict(pid=process.pid,root=str(root),cwd=str(release),argv=command,log=str(log))))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release',required=True)
    args=parser.parse_args();source=payload(args.release);compile(source,'<authorized remote dispatcher>','exec')
    result=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
        'N607','/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input=source.encode('utf-8'),capture_output=True,timeout=30)
    print(result.stdout.decode('utf-8'));print(result.stderr.decode('utf-8',errors='replace'))
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
