"""Phase1 source-only fixed phase candidate exploration; no model fitting."""
import hashlib,json,subprocess
from pathlib import Path

source_path=Path('E:/type10-7/github_publish/CVS-RFFI-repo/experiments/pamf_phase1_flow/phase_orbit.py')
source=source_path.read_text(encoding='utf8');digest=hashlib.sha256(source.encode()).hexdigest()
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
import json,math,numpy as np,torch
from experiments.pamf_phase1_flow import experiment as core,anchor_balanced as a
from experiments.pamf_phase1_flow.dura_source_trisig_diagnostic import PARENT
from experiments.pamf_phase1_flow.feature_conditioned_experiment import source_labels
exec(__PHASE_SOURCE__,globals())
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract);dura=a.frozen_dura(root,'cpu')
assert a.sha(root/a.DURA)=='14200ac0f8eef01f57aa82e8fb5122ff855fc81208983d9234d6c5caef2e28ee'
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
fixed=np.concatenate([np.flatnonzero(roles==role)[np.linspace(0,count-1,n,dtype=int)] for role,count,n in (('L_s',6300,128),('U_s',56700,384))])
phases=(0.,math.pi/8,-math.pi/8,math.pi/4,-math.pi/4,math.pi/2,-math.pi/2,math.pi)
rules=('soft_mean','majority','max_confidence');results={};states={}
with torch.no_grad():
 for view in ('clean_raw','clean_dura','01','20'):
  if view not in ('clean_raw','clean_dura'):
   with np.load(root/'runs'/PARENT/f'DURA_default-{int(view)}.npz',allow_pickle=False) as z:
    saved={k:z[k] for k in ('source_positions','physical_ids','roles','clean','native','state')}
   assert np.array_equal(saved['source_positions'],fixed) and np.array_equal(saved['physical_ids'],ids[fixed])
   assert np.array_equal(saved['roles'],roles[fixed]) and np.array_equal(saved['clean'],clean[fixed])
   raw=np.load(cache/'epochs'/view/'received.npy',mmap_mode='r',allow_pickle=False);states[view]=saved['state']
  collected={k:[] for k in ('base',*rules)};changed={k:[] for k in rules}
  for start in range(0,len(fixed),32):
   ix=fixed[start:start+32];received=core.tensor(clean[ix] if view in ('clean_raw','clean_dura') else raw[ix],'cpu')
   anchor=received if view=='clean_raw' else dura(received)
   if view not in ('clean_raw','clean_dura'):torch.testing.assert_close(anchor,core.tensor(saved['native'][start:start+len(ix)],'cpu'),rtol=1e-4,atol=1e-5)
   candidates=phase_candidates(anchor,phases);torch.testing.assert_close(candidates[:,0],anchor,rtol=0,atol=0)
   torch.testing.assert_close(candidates.square().sum((2,3)),anchor.square().sum((1,2))[:,None].expand(-1,len(phases)),rtol=1e-5,atol=1e-4)
   flat=candidates.reshape(-1,2,256);logits=teacher.classify_features(teacher.features(flat)).reshape(len(ix),len(phases),6)
   assert torch.isfinite(logits).all();pred=logits.argmax(2);collected['base'].append(pred[:,0].cpu())
   for rule in rules:
    chosen=select_phase(logits,rule=rule);collected[rule].append(pred.gather(1,chosen[:,None]).squeeze(1).cpu());changed[rule].append((chosen!=0).to(torch.int64).cpu())
  results[view]={k:np.from_dlpack(torch.cat(v)) for k,v in collected.items()}
  results[view].update({k+'_changed':np.from_dlpack(torch.cat(v)) for k,v in changed.items()})
  print('PHASE_PREDICTIONS_FIXED',view,flush=True)
# Source labels are joined only after all candidate decisions are fixed.
labels=source_labels(root,cache,clean,roles,contract);ref=results['clean_raw']['base'];labeled=roles[fixed]=='L_s'
assert np.all(labels[fixed[128:]]==-1)
out=[]
def counts(reference,before,after,mask):
 a=before==reference;b=after==reference
 return dict(count=int(mask.sum()),before=int((mask&a).sum()),after=int((mask&b).sum()),repairs=int((mask&~a&b).sum()),damages=int((mask&a&~b).sum()))
for view in ('clean_raw','clean_dura','01','20'):
 z=results[view];masks=[('all',np.ones(len(fixed),dtype=bool)),('L_s',labeled)]
 if view not in ('clean_raw','clean_dura'):masks.append(('blocked',states[view]=='BLOCKED'))
 rows=[]
 for group,mask in masks:
  for rule in rules:
   rows.append(dict(group=group,rule=rule,rotated=int((mask&(z[rule+'_changed']!=0)).sum()),decision_changes=int((mask&(z[rule]!=z['base'])).sum()),
    clean_teacher=counts(ref,z['base'],z[rule],mask),L_s_truth=counts(labels[fixed],z['base'],z[rule],mask&labeled)))
 out.append(dict(view=view,groups=rows))
print('PHASE_PROBE_RESULT '+json.dumps(dict(status='SOURCE_ONLY_PHASE_ORBIT_PROBE',source_records=512,source_views=['clean_raw','clean_dura',1,20],
 phases_rad=list(phases),rules=list(rules),source_code_sha256=__SOURCE_HASH__,F_run=core.F_RUN,F_row=core.F_ROW,
 source_L_s_labels_materialized=6300,source_L_s_labels_scored=128,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,
 all_candidate_predictions_before_source_labels=True,clean_raw_bypasses_DURA=True,clean_dura_uses_DURA=True,results=out,
 caveat='Source training records, not independent test. Clean teacher agreement differs from TX accuracy; phase rotations preserve energy but may change IQ fidelity.'),ensure_ascii=False),flush=True)
'''.replace('__PHASE_SOURCE__',repr(source)).replace('__SOURCE_HASH__',repr(digest))
remote=remote.replace('__INNER_SCRIPT__',repr(inner))
proc=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=320)
stdout=proc.stdout.decode('utf8',errors='replace');print(stdout);print(proc.stderr.decode('utf8',errors='replace'))
if proc.returncode==0:
 line=next(s[len('PHASE_PROBE_RESULT '):] for s in stdout.splitlines() if s.startswith('PHASE_PROBE_RESULT '))
 path=Path(__file__).parent/'result.json'
 if path.exists():
  if json.loads(path.read_text(encoding='utf8'))!=json.loads(line):raise ValueError('Repeated source result differs from saved evidence')
  print('VERIFIED_IDENTICAL_SAVED_EVIDENCE')
 else:
  with path.open('x',encoding='utf8') as f:f.write(line+'\n')
raise SystemExit(proc.returncode)
