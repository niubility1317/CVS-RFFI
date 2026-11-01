import ast,subprocess,json
from pathlib import Path
p=Path(__file__).parent
t=ast.parse((p/'phase1_source_phase_utility_probe.py').read_text(encoding='utf8'))
i=ast.literal_eval(next(n.value for n in t.body if isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id=='inner' for z in n.targets)))
a=i.index('q_rng=np.random.default_rng(2026110251)');b=i.index('fixed=np.concatenate(groups)',a)
i=i[:a]+'''rng=np.random.default_rng(2026110501);groups=[]\nfor role,take in (("L_s",1024),("U_s",3072)):\n available=np.flatnonzero(roles==role);groups.append(np.sort(rng.choice(available,size=take,replace=False)))\n'''+i[b:]
i=i.split("views=('clean','practical_low_suburban'",1)[0]
i+="\nprint('MEMORY_ID_AUDIT '+json.dumps(dict(total=int(len(fixed)),source_unique_physical=int(len(np.unique(ids[fixed]))),train_val_overlap=int(len(np.intersect1d(ids[fixed[split==0]],ids[fixed[split==1]]))),train_holdout_overlap=int(len(np.intersect1d(ids[fixed[split==0]],ids[fixed[split==2]]))),validation_holdout_overlap=int(len(np.intersect1d(ids[fixed[split==1]],ids[fixed[split==2]]))))))\n"
compile(i,'<memory-id-audit>','exec')
r='''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=180)
'''.replace('INNER',repr(i),1)
x=subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607','python3','-'],input=r.encode(),capture_output=True,timeout=200)
print(x.stdout.decode('utf8',errors='replace'));print(x.stderr.decode('utf8',errors='replace'))
if x.returncode:raise SystemExit(x.returncode)
line=next(z for z in x.stdout.decode().splitlines() if z.startswith('MEMORY_ID_AUDIT '));result=json.loads(line[len('MEMORY_ID_AUDIT '):]);assert result['train_val_overlap']==result['train_holdout_overlap']==result['validation_holdout_overlap']==0
(p/'phase1_source_ground_memory_id_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print('VERIFIED physical-id separation')
