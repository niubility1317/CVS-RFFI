"""Compare source smoke routing on CPU against the GPU worker's recorded choices."""
import subprocess

remote=r'''
from pathlib import Path
import json,os,subprocess
r=Path('/home/szu2070436088/2510044040/CV-SincNet');run='20261101-phase1-multi-utility-router-s2026092701-r01';out=r/'runs'/run
gpu=out/'gpu_smoke.json'
if not gpu.exists():print('WAITING_FOR_GPU_SOURCE_SMOKE');raise SystemExit(0)
release=r/'releases'/(run+'-release');native=r/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='9404469fe2fbad3ccc247a11eaf5d542899dd4e5'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
inner=r"""
from pathlib import Path
import json,numpy as np,torch
from experiments.pamf_phase1_flow import experiment as core,multi_utility_experiment as g
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
models=g.load_models(root,'cpu');cache,clean,roles=core.data(root,models[1]);ix=np.flatnonzero((roles=='L_s')|(roles=='U_s'))[:128]
raw=np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False);_,_,choice=g.route(models,core.tensor(raw[ix],'cpu'))
counts=[int((choice==i).sum().item()) for i in range(4)]
gpu=json.loads((root/'runs/20261101-phase1-multi-utility-router-s2026092701-r01/gpu_smoke.json').read_text())
assert gpu['status']=='PASS' and gpu['source_rows']==128 and gpu['choice_counts']==counts
assert gpu['asset_sha256']==g.ASSET_SHA and gpu['TX_labels_read']==gpu['V_reads']==gpu['test_IQ_reads']==0
print('VERIFIED_CPU_GPU_SOURCE_CHOICE_COUNTS',counts,'source_rows',len(ix),'labels_read',0)
"""
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-c',inner],cwd=release,env=env,check=True,timeout=90)
'''
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=110)
print(p.stdout.decode('utf8',errors='replace'));print(p.stderr.decode('utf8',errors='replace'));raise SystemExit(p.returncode)
