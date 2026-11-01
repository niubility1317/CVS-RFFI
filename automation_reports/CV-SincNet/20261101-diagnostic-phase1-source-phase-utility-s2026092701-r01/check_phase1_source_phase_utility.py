import ast,subprocess
from pathlib import Path
p=Path(__file__).parent;s=(p/'phase1_source_phase_utility_probe.py').read_text(encoding='utf8');tree=ast.parse(s)
node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='inner' for x in n.targets) and isinstance(n.value,ast.Constant));inner=ast.literal_eval(node)
old='assert all(int((split==i).sum())>200 for i in range(3))'
new=old+"\nselection=np.concatenate((np.flatnonzero(split==0)[:2],np.flatnonzero(split==1)[:1],np.flatnonzero(split==2)[:1]));fixed=fixed[selection];source_rx=source_rx[selection];source_day=source_day[selection];split=split[selection];assert len(fixed)==4"
assert inner.count(old)==1;inner=inner.replace(old,new)
marker="print('ALL_PHASE_UTILITY_SOURCE_CANDIDATES_FIXED_BEFORE_SOURCE_LABELS',flush=True)";assert inner.count(marker)==1;inner=inner.split(marker,1)[0]+marker+"\nprint('VERIFIED_PHASE_UTILITY_SOURCE_SMOKE',len(fixed),'test_reads',0,flush=True)\n"
remote='''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=300)
'''.replace('INNER',repr(inner),1)
proc=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=320)
print(proc.stdout.decode('utf8',errors='replace'));print(proc.stderr.decode('utf8',errors='replace'));raise SystemExit(proc.returncode)
