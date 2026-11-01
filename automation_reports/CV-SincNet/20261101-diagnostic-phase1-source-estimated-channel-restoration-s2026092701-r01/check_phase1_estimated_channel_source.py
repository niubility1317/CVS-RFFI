import ast,json,subprocess
from pathlib import Path
p=Path(__file__).parent;s=(p/'phase1_source_estimated_channel_restoration_probe.py').read_text(encoding='utf8');tree=ast.parse(s)
node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='inner' for x in n.targets) and isinstance(n.value,ast.Constant));inner=ast.literal_eval(node)
old='keep=split>0;fixed=fixed[keep];source_rx=source_rx[keep];source_day=source_day[keep];split=split[keep];assert len(fixed)>300'
new='selection=np.concatenate((np.flatnonzero(split==1)[:2],np.flatnonzero(split==2)[:2]));fixed=fixed[selection];source_rx=source_rx[selection];source_day=source_day[selection];split=split[selection];assert len(fixed)==4'
assert inner.count(old)==1;inner=inner.replace(old,new)
marker="print('ALL_ESTIMATED_CHANNEL_SOURCE_CANDIDATES_FIXED_BEFORE_SOURCE_LABELS',flush=True)";assert inner.count(marker)==1;inner=inner.split(marker,1)[0]+marker+"\nprint('VERIFIED_ESTIMATED_CHANNEL_SOURCE_SMOKE',len(fixed),'test_reads',0,flush=True)\n"
d=json.loads((p/'phase1_source_channel_estimator_probe.json').read_text(encoding='utf8'));payload=json.dumps({'model_state':d['model_state']})
remote='''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
inner=__INNER__;payload=__PAYLOAD__
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',inner],cwd=release,env=env,input=payload,text=True,check=True,timeout=300)
'''.replace('__INNER__',repr(inner)).replace('__PAYLOAD__',repr(payload))
proc=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=320)
print(proc.stdout.decode('utf8',errors='replace'));print(proc.stderr.decode('utf8',errors='replace'));raise SystemExit(proc.returncode)
