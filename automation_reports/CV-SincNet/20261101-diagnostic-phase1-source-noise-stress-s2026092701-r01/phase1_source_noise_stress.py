"""Diagnose expert complementarity under controlled added noise, source only."""
import subprocess
from pathlib import Path

remote = r'''
from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet')
release=root/'releases/20261101-phase1-boundary-student-E30-s2026092701-r02-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='f72f9618859fce30a1983a3b4b10c8c869b2dbba'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',__INNER_SCRIPT__],cwd=release,env=env,check=True,timeout=600)
'''
inner = r'''
from pathlib import Path
import hashlib,json,numpy as np,torch
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,anchor_balanced as a,boundary_student_experiment as bs,feature_conditioned_experiment as fs
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract);dura=a.frozen_dura(root,'cpu');digest=a.sha(root/a.DURA)
assert digest=='14200ac0f8eef01f57aa82e8fb5122ff855fc81208983d9234d6c5caef2e28ee'
bp=root/'runs/20261101-phase1-boundary-student-E30-s2026092701-r02/flow/final.pt';fp=root/'runs/20261101-phase1-feature-conditioned-E30-s2026092701-r02/flow/final.pt'
assert hashlib.sha256(bp.read_bytes()).hexdigest()=='951bd2dc49753db8b21af0cba6afcf0124ea05f43d59203957a0dd603b8eb0a7'
assert hashlib.sha256(fp.read_bytes()).hexdigest()=='4b66b90ced7ba4bc31bf7bad1b49483473d9b42b139a4faef2676bbbeaa4b8f9'
commit='f72f9618859fce30a1983a3b4b10c8c869b2dbba';boundary=bs.load(bp,'cpu',commit,digest);feature=fs.load(fp,'cpu',commit,digest)
groups=[]
for role,count,old_n,n in (('L_s',6300,128,1024),('U_s',56700,384,3072)):
 available=np.flatnonzero(roles==role);prior=available[np.linspace(0,count-1,old_n,dtype=int)]
 available=np.setdiff1d(available,prior,assume_unique=True);groups.append(available[np.linspace(0,len(available)-1,n,dtype=int)])
fixed=np.concatenate(groups);order=np.random.default_rng(2026110150).permutation(len(fixed));ix=fixed[np.sort(order[3276:])][:512]
assert len(ix)==len(np.unique(ix))==512
views={}
with torch.no_grad():
 for view in ('clean','01','20'):
  if view!='clean':raw=np.load(cache/'epochs'/view/'received.npy',mmap_mode='r',allow_pickle=False)
  for snr in ([None] if view=='clean' else [None,15,5]):
   name=view if snr is None else view+'_plus_'+str(snr)+'dB';generator=torch.Generator(device='cpu').manual_seed(2026110160+(0 if view=='clean' else int(view)*100)+(snr or 0))
   probs=[]
   for start in range(0,len(ix),32):
    ids=ix[start:start+32];x=core.tensor(clean[ids] if view=='clean' else raw[ids],'cpu')
    if snr is not None:
     rms=x.square().mean((1,2),keepdim=True).sqrt();noise=torch.randn(x.shape,generator=generator,dtype=x.dtype)*rms*(10**(-snr/20));x=x+noise
     x=x*(rms/x.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-12))
    anchor=dura(x);feat=teacher.features(anchor);base=teacher.classify_features(feat)
    outputs=(x,anchor,boundary(x,anchor),feature(x,anchor,feat,base))
    value=torch.stack([F.softmax(teacher.classify_features(teacher.features(y)),1) for y in outputs],1)
    assert torch.isfinite(value).all();probs.append(value.cpu())
   views[name]=np.from_dlpack(torch.cat(probs));print('SOURCE_VIEW_OUTPUTS_FIXED',name,flush=True)
print('ALL_SOURCE_OUTPUTS_FIXED',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[ix]=='L_s';assert np.all(labels[ix[~labeled]]==-1)
ref=views['clean'][:,0,:];target=np.where(labeled,labels[ix],ref.argmax(1));valid=labeled|(ref.max(1)>=.95);results=[]
for name,p in views.items():
 pred=p.argmax(2);correct=pred==target[:,None];base=correct[:,1];three=correct[:,1:].any(1);four=correct.any(1)
 row=dict(view=name,valid=int(valid.sum()),labeled=int((valid&labeled).sum()),pseudo=int((valid&~labeled).sum()),
          raw=int((valid&correct[:,0]).sum()),dura=int((valid&base).sum()),boundary=int((valid&correct[:,2]).sum()),feature=int((valid&correct[:,3]).sum()),
          oracle_three=int((valid&three).sum()),oracle_four=int((valid&four).sum()),raw_unique=int((valid&~three&correct[:,0]).sum()),
          baseline_wrong_fixable=int((valid&~base&four).sum()))
 results.append(row)
print('SOURCE_NOISE_STRESS '+json.dumps(dict(status='SOURCE_ONLY_STRESS',ids=len(ix),results=results,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,
 caveat='Added white Gaussian noise is a synthetic source stress and may not model satellite channels. Experts trained on source IDs. Mixed real L_s and confident clean-teacher U_s pseudo-targets. Oracle is inaccessible.'),ensure_ascii=False),flush=True)
'''
remote = remote.replace('__INNER_SCRIPT__', repr(inner))
proc = subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'], input=remote.encode(), capture_output=True, timeout=620)
stdout=proc.stdout.decode('utf-8',errors='replace'); print(stdout); print(proc.stderr.decode('utf-8',errors='replace'))
if proc.returncode==0:
    line=next(s[len('SOURCE_NOISE_STRESS '):] for s in stdout.splitlines() if s.startswith('SOURCE_NOISE_STRESS '))
    with (Path(__file__).parent/'phase1_source_noise_stress.json').open('x',encoding='utf-8') as out: out.write(line+'\n')
raise SystemExit(proc.returncode)
