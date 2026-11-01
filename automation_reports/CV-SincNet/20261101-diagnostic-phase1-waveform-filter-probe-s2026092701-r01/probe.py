"""Source-only held-out-ID ridge probe; all learned inputs are inference-available."""
import json
import subprocess
from pathlib import Path

remote=r'''
from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet')
release=root/'releases/20261101-phase1-filter-student-E30-s2026092701-r01-release'
native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b0c93a3aaac27deebaf2e751d42523f485bfd17d'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
inner=r"""
from pathlib import Path
import json,numpy as np,torch
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,anchor_balanced as a
from experiments.pamf_phase1_flow.filter_student import apply_filter_delta,project_delta
from experiments.pamf_phase1_flow.source_fir_oracle import fit_paired_fir
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract);dura=a.frozen_dura(root,'cpu')
assert a.sha(root/a.DURA)=='14200ac0f8eef01f57aa82e8fb5122ff855fc81208983d9234d6c5caef2e28ee'
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
groups=[]
for role,count,n in (('L_s',6300,256),('U_s',56700,768)):
 all_ix=np.flatnonzero(roles==role);old_ix=all_ix[np.linspace(0,count-1,n,dtype=int)]
 remaining=np.setdiff1d(all_ix,old_ix,assume_unique=True);groups.append(remaining[np.linspace(0,len(remaining)-1,n,dtype=int)])
selected=np.concatenate(groups)
assert len(selected)==len(np.unique(selected))==1024
rng=np.random.default_rng(2026110120);order=rng.permutation(len(selected));split=np.empty(len(selected),dtype=np.int64)
split[order[:614]]=0;split[order[614:819]]=1;split[order[819:]]=2
assert [int((split==k).sum()) for k in range(3)]==[614,205,205]
records={k:[] for k in ('features','coefficient','anchor','clean','reference','old','split','view')}
with torch.no_grad():
 for view in (1,20):
  received=np.load(cache/'epochs'/f'{view:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
  for start in range(0,len(selected),32):
   ix=selected[start:start+32];x=core.tensor(received[ix],'cpu');target=core.tensor(clean[ix],'cpu');anchor=dura(x)
   f=teacher.features(anchor);logits=teacher.classify_features(f);base=logits.argmax(1)
   ref=teacher.classify_features(teacher.features(target)).argmax(1)
   energy_x=x.square().mean((1,2)).clamp_min(1e-8);energy_a=anchor.square().mean((1,2)).clamp_min(1e-8)
   xx=torch.complex(x[:,0],x[:,1]);aa=torch.complex(anchor[:,0],anchor[:,1])
   cross=(xx*aa.conj()).mean(1)/(xx.abs().square().mean(1)*aa.abs().square().mean(1)).sqrt().clamp_min(1e-8)
   features=torch.cat((F.normalize(f,dim=1),F.softmax(logits,dim=1),energy_x.log()[:,None],energy_a.log()[:,None],cross.real[:,None],cross.imag[:,None]),1)
   w=F.pad(aa,(4,4)).unfold(1,9,1);design=torch.cat((w,w.conj()),2);h=design.conj().transpose(1,2)
   gram=h@design;scale=gram.diagonal(dim1=-2,dim2=-1).real.mean(-1).clamp_min(1e-8)
   eye=torch.eye(18,dtype=gram.dtype);truth=torch.complex(target[:,0],target[:,1])
   coeff=torch.linalg.solve(gram+1e-4*scale[:,None,None]*eye,h@truth.unsqueeze(-1)).squeeze(-1)
   coeff[:,4]-=1
   vector=torch.stack((coeff.real,coeff.imag),dim=-1).reshape(len(ix),2,9,2)
   delta=apply_filter_delta(anchor,vector);ratio=delta.flatten(1).norm(dim=1)/anchor.flatten(1).norm(dim=1).clamp_min(1e-8)
   vector=vector*(.2/ratio.clamp_min(1e-12)).clamp(max=1)[:,None,None,None]
   expected=project_delta(anchor,fit_paired_fir(anchor,target,taps=9,widely_linear=True,ridge=1e-4),epsilon=.2)
   torch.testing.assert_close(anchor+apply_filter_delta(anchor,vector),expected,rtol=1e-4,atol=2e-5)
   for key,v in [('features',features),('coefficient',vector.reshape(len(ix),36)),('anchor',anchor),('clean',target),('reference',ref),('old',base)]:records[key].append(np.from_dlpack(v.cpu().contiguous()))
   records['split'].append(split[start:start+len(ix)]);records['view'].append(np.full(len(ix),view))
  print('SOURCE_FEATURES_READY',view,flush=True)
data={k:np.concatenate(v,axis=0) for k,v in records.items()}
assert data['features'].shape==(2048,170) and all(np.array_equal(data['split'][:1024],data['split'][1024:]) for _ in (0,))
train=data['split']==0;validation=data['split']==1;test=data['split']==2
mu=data['features'][train].mean(0);sd=data['features'][train].std(0).clip(min=1e-4)
features=np.clip((data['features']-mu)/sd,-10,10).astype(np.float32)
features=np.column_stack((np.ones(len(features),dtype=np.float32),features)).astype(np.float32)
feat=core.tensor(features,'cpu');anchor_all=core.tensor(data['anchor'],'cpu');clean_all=core.tensor(data['clean'],'cpu')
tr=np.flatnonzero(train);va=np.flatnonzero(validation)
def make_wave(index,weight):
 index=torch.tensor(index.tolist(),dtype=torch.long) if isinstance(index,np.ndarray) else index
 anchor=anchor_all[index];coeff=(feat[index]@weight).reshape(-1,2,9,2)
 return project_delta(anchor,anchor+apply_filter_delta(anchor,coeff),epsilon=.2)
def score(index,weight):
 with torch.no_grad():
  y=make_wave(index,weight);truth=clean_all[torch.tensor(index.tolist(),dtype=torch.long)]
  energy=truth.square().mean((1,2)).clamp_min(1e-4)
  return float(((y-truth).square().mean((1,2))/energy).mean())
zero=torch.zeros(171,36);base_score=score(va,zero)
best_score=base_score;best_weight=zero.clone();best_config={'kind':'identity','validation_iq_nmse':base_score};records_score=[]
for regularization in (0.,.01,.1):
 torch.manual_seed(2026110121);weight=torch.nn.Parameter(torch.zeros(171,36));opt=torch.optim.Adam([weight],lr=.0001)
 rng=np.random.default_rng(2026110122)
 for epoch in range(1,17):
  shuffled=rng.permutation(tr)
  for start in range(0,len(shuffled),128):
   idx=shuffled[start:start+128];wave=make_wave(idx,weight);truth=clean_all[torch.tensor(idx.tolist(),dtype=torch.long)]
   energy=truth.square().mean((1,2)).clamp_min(1e-4)
   loss=((wave-truth).square().mean((1,2))/energy).mean()+regularization*weight[1:].square().sum()
   if not torch.isfinite(loss):raise FloatingPointError('Finite source waveform loss required')
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([weight],1.,error_if_nonfinite=True);opt.step()
  if epoch in (2,4,8,16):
   val=score(va,weight);records_score.append(dict(l2=regularization,epoch=epoch,validation_iq_nmse=val))
   if val<best_score:
    best_score=val;best_weight=weight.detach().clone();best_config=dict(kind='waveform_loss',l2=regularization,epoch=epoch,validation_iq_nmse=val)
 print('WAVEFORM_SOURCE_FIT_READY',regularization,flush=True)
with torch.no_grad():y=np.from_dlpack(make_wave(np.flatnonzero(test),best_weight).cpu().contiguous())
truth=data['clean'][test];anchor=data['anchor'][test]
def nmse(y,truth):return float(np.mean(np.mean((y-truth)**2,axis=(1,2))/np.maximum(np.mean(truth**2,axis=(1,2)),1e-4)))
candidate=[];oracle=[]
with torch.no_grad():
 for offset in range(0,len(y),32):
  sl=slice(offset,offset+32);new=core.tensor(y[sl],'cpu');old=core.tensor(anchor[sl],'cpu');clean_b=core.tensor(truth[sl],'cpu')
  fit=project_delta(old,fit_paired_fir(old,clean_b,taps=9,widely_linear=True,ridge=1e-4),epsilon=.2)
  candidate.append(np.from_dlpack(teacher.classify_features(teacher.features(new)).argmax(1).cpu().contiguous()))
  oracle.append(np.from_dlpack(teacher.classify_features(teacher.features(fit)).argmax(1).cpu().contiguous()))
candidate=np.concatenate(candidate);oracle=np.concatenate(oracle);ref=data['reference'][test];old=data['old'][test];views=data['view'][test]
results=[]
for view in (1,20):
 mask=views==view;before=old==ref;after=candidate==ref;upper=oracle==ref
 results.append(dict(view=view,count=int(mask.sum()),baseline_teacher_match=int((mask&before).sum()),
  probe_teacher_match=int((mask&after).sum()),oracle_teacher_match=int((mask&upper).sum()),
  repairs=int((mask&~before&after).sum()),damages=int((mask&before&~after).sum()),
  baseline_iq_nmse=nmse(anchor[mask],truth[mask]),probe_iq_nmse=nmse(y[mask],truth[mask])))
print('PROBE_RESULT '+json.dumps(dict(status='SOURCE_ONLY_HELDOUT_ID_WAVEFORM_PROBE',source_ids=1024,train_ids=614,validation_ids=205,heldout_ids=205,
 disjoint_from_previous_probe=True,feature_count=170,views=[1,20],candidate_settings=records_score,selected=best_config,results=results,
 TX_labels_read=0,V_reads=0,test_IQ_reads=0,clean_used_for_train_and_validation_targets=True,
 caveat='Held-out only from this source probe; source records belong to overall project training. Clean teacher agreement is not TX accuracy.'),ensure_ascii=False),flush=True)
"""
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',inner],cwd=release,env=env,check=True,timeout=500)
'''
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=520)
stdout=p.stdout.decode('utf8',errors='replace');print(stdout);print(p.stderr.decode('utf8',errors='replace'))
if p.returncode==0:
 line=next(x[len('PROBE_RESULT '):] for x in stdout.splitlines() if x.startswith('PROBE_RESULT '))
 path=Path(__file__).parent/'phase1_source_waveform_filter_probe.json'
 if path.exists():
  if json.loads(path.read_text(encoding='utf8'))!=json.loads(line):raise ValueError('Repeated source result differs from saved evidence')
  print('VERIFIED_IDENTICAL_SAVED_EVIDENCE')
 else:
  with path.open('x',encoding='utf8') as f:f.write(line+'\n')
raise SystemExit(p.returncode)
