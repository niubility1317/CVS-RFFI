"""Read-only source clean half-window cross-fit diagnostic, not a receiver."""
import hashlib,json,subprocess
from pathlib import Path

source_path=Path('E:/type10-7/github_publish/CVS-RFFI-repo/experiments/pamf_phase1_flow/source_fir_crossfit.py')
source=source_path.read_text(encoding='utf8')
digest=hashlib.sha256(source.encode()).hexdigest()
remote=r'''
from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet')
release=root/'releases/20261101-phase1-filter-student-E30-s2026092701-r01-release'
native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
inner=__INNER_SCRIPT__
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',inner],cwd=release,env=env,check=True,timeout=300)
'''
inner=r'''
from pathlib import Path
import json,numpy as np,torch
from experiments.pamf_phase1_flow import experiment as core,anchor_balanced as a
from experiments.pamf_phase1_flow.dura_source_trisig_diagnostic import PARENT
from experiments.pamf_phase1_flow.source_fir_oracle import fit_paired_fir
from experiments.pamf_phase1_flow.filter_student import project_delta
exec(__FIR_SOURCE__,globals())
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract);dura=a.frozen_dura(root,'cpu')
assert a.sha(root/a.DURA)=='14200ac0f8eef01f57aa82e8fb5122ff855fc81208983d9234d6c5caef2e28ee'
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
groups=[]
for role,count,n in (('L_s',6300,256),('U_s',56700,768)):
 available=np.flatnonzero(roles==role)
 for unused in range(3):
  prior=available[np.linspace(0,len(available)-1,n,dtype=int)]
  available=np.setdiff1d(available,prior,assume_unique=True)
 groups.append(available[np.linspace(0,len(available)-1,n,dtype=int)])
fixed=np.concatenate(groups)
assert len(fixed)==len(np.unique(fixed))==1024
ridges=(('r0001',.0001),('r001',.001),('r01',.01),('r1',.1),('r10',1.))
methods=['dura']+[f'{kind}9_{name}' for kind in ('cross','full') for name,ridge in ridges]
results=[]
def clip_halfwise(anchor,y):
 return torch.cat([project_delta(anchor[:,:,sl],y[:,:,sl],epsilon=.2) for sl in (slice(0,128),slice(128,256))],dim=2)
with torch.no_grad():
 for view in (1,20):
  ix=fixed
  assert len(np.unique(ids[ix]))==1024
  raw=np.load(cache/'epochs'/f'{view:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
  scores={k:[] for k in methods};errors={k:[] for k in methods};reference=[]
  for start in range(0,len(ix),32):
   sl=slice(start,start+32);received=core.tensor(raw[ix[sl]],'cpu');truth=core.tensor(clean[ix[sl]],'cpu');anchor=dura(received)
   ref=teacher.classify_features(teacher.features(truth)).argmax(1);reference.append(ref.cpu())
   values={'dura':anchor}
   for name,ridge in ridges:
    residual=truth-anchor
    cross=crossfit_paired_fir(anchor,residual,taps=9,ridge=ridge)
    full=fit_paired_fir(anchor,residual,taps=9,widely_linear=True,ridge=ridge)
    values[f'cross9_{name}']=clip_halfwise(anchor,anchor+cross)
    values[f'full9_{name}']=clip_halfwise(anchor,anchor+full)
   energy=truth.square().mean((1,2)).clamp_min(1e-4)
   for name,wave in values.items():
    y=teacher.classify_features(teacher.features(wave));scores[name].append(y.argmax(1).cpu())
    errors[name].append(((wave-truth).square().mean((1,2))/energy).cpu())
  pred={k:np.from_dlpack(torch.cat(v)) for k,v in scores.items()};err={k:torch.cat(v) for k,v in errors.items()}
  ref=np.from_dlpack(torch.cat(reference));baseline=pred['dura']==ref;groups=[]
  for group,mask in [('all',np.ones(len(ix),dtype=bool)),('L_s',roles[ix]=='L_s'),('U_s',roles[ix]=='U_s')]:
   selected=torch.tensor(mask.tolist(),dtype=torch.bool)
   for name in methods:
    new=pred[name]==ref
    groups.append(dict(group=group,method=name,count=int(mask.sum()),teacher_matches=int((mask&new).sum()),
     repairs=int((mask&~baseline&new).sum()),damages=int((mask&baseline&~new).sum()),mean_iq_nmse=float(err[name][selected].mean())))
  results.append(dict(view=view,groups=groups));print('SOURCE_VIEW_COMPLETE',view,flush=True)
print('CROSSFIT_RESULT '+json.dumps(dict(status='SOURCE_ONLY_FRESH_RESIDUAL_SWEEP',source_records=1024,views=[1,20],taps=[9],ridges=[.0001,.001,.01,.1,1.],
 source_code_sha256=__SOURCE_HASH__,F_run=core.F_RUN,F_row=core.F_ROW,TX_labels_read=0,V_reads=0,test_IQ_reads=0,
 halfwise_radius=.2,clean_half_used_to_fit_other_half=True,
 caveat='Cross-fit remains privileged and noncausal; source training records, shared raw input across half boundary; teacher match is not TX accuracy.',results=results),ensure_ascii=False),flush=True)
'''.replace('__FIR_SOURCE__',repr(source)).replace('__SOURCE_HASH__',repr(digest))
remote=remote.replace('__INNER_SCRIPT__',repr(inner))
proc=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=320)
stdout=proc.stdout.decode('utf8',errors='replace');print(stdout);print(proc.stderr.decode('utf8',errors='replace'))
if proc.returncode==0:
 line=next(s[len('CROSSFIT_RESULT '):] for s in stdout.splitlines() if s.startswith('CROSSFIT_RESULT '))
 path=Path(__file__).parent/'fresh_result.json'
 if path.exists():
  if json.loads(path.read_text(encoding='utf8'))!=json.loads(line):raise ValueError('Repeated source result differs from saved evidence')
  print('VERIFIED_IDENTICAL_SAVED_EVIDENCE')
 else:
  with path.open('x',encoding='utf8') as f:f.write(line+'\n')
raise SystemExit(proc.returncode)
