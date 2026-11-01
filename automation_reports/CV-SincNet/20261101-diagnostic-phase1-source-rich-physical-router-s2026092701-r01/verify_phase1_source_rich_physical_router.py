"""Replay frozen source router and experts on held-out IDs, without refitting."""
import json,subprocess
from pathlib import Path

p=Path(__file__).parent
result=json.loads((p/'phase1_source_rich_physical_router_probe.json').read_text(encoding='utf8'))
assert result['selected']['kind']=='source_router' and result['router_state'] is not None
payload={k:result[k] for k in ('selected','router_state','feature_mean','feature_std','heldout','boundary_checkpoint_sha256','feature_checkpoint_sha256')}
remote=r'''
from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet')
release=root/'releases/20261101-phase1-boundary-student-E30-s2026092701-r02-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
inner=__INNER_SCRIPT__
payload=__PAYLOAD__
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',inner],cwd=release,env=env,input=payload,text=True,check=True,timeout=600)
'''
inner=r'''
from pathlib import Path
import hashlib,json,sys,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,anchor_balanced as a,boundary_student_experiment as bs,feature_conditioned_experiment as fs
expected=json.load(sys.stdin);root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract);dura=a.frozen_dura(root,'cpu')
digest=a.sha(root/a.DURA);assert digest=='14200ac0f8eef01f57aa82e8fb5122ff855fc81208983d9234d6c5caef2e28ee'
bp=root/'runs/20261101-phase1-boundary-student-E30-s2026092701-r02/flow/final.pt';fp=root/'runs/20261101-phase1-feature-conditioned-E30-s2026092701-r02/flow/final.pt'
assert hashlib.sha256(bp.read_bytes()).hexdigest()==expected['boundary_checkpoint_sha256'] and hashlib.sha256(fp.read_bytes()).hexdigest()==expected['feature_checkpoint_sha256']
commit='f72f9618859fce30a1983a3b4b10c8c869b2dbba';boundary=bs.load(bp,'cpu',commit,digest);feature=fs.load(fp,'cpu',commit,digest)
groups=[]
for role,count,old_n,n in (('L_s',6300,128,1024),('U_s',56700,384,3072)):
 available=np.flatnonzero(roles==role);prior=available[np.linspace(0,count-1,old_n,dtype=int)]
 available=np.setdiff1d(available,prior,assume_unique=True)
 for _ in range(3):
  old=available[np.linspace(0,len(available)-1,n,dtype=int)];available=np.setdiff1d(available,old,assume_unique=True)
 take=128 if role=='L_s' else 384;prior_stress=available[np.linspace(0,len(available)-1,take,dtype=int)]
 available=np.setdiff1d(available,prior_stress,assume_unique=True);old4=available[np.linspace(0,len(available)-1,n,dtype=int)]
 available=np.setdiff1d(available,old4,assume_unique=True);groups.append(available[np.linspace(0,len(available)-1,n,dtype=int)])
fixed=np.concatenate(groups);order=np.random.default_rng(2026110201).permutation(len(fixed));ix=fixed[np.sort(order[3276:])]
assert len(ix)==len(np.unique(ix))==820
router=nn.Sequential(nn.Linear(len(expected['feature_mean']),128),nn.GELU(),nn.Linear(128,4))
router.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in expected['router_state'].items()},strict=True);router.eval().requires_grad_(False)
mean=np.asarray(expected['feature_mean'],dtype=np.float32);std=np.asarray(expected['feature_std'],dtype=np.float32)
threshold=expected['selected']['threshold'];experts=('raw','dura','boundary','feature');views={}
view_names=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban')
from leo_practical.channel import Config,receiver_for_session,stable_seed,normalize_input
from leo_practical.residual import ResidualChannel
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);rx=np.load(cache/'train/rx.npy',allow_pickle=False);day=np.load(cache/'train/day.npy',allow_pickle=False)
namespace='phase1-source-physical-router-2026110202'
with torch.no_grad():
 for view in view_names:
  if view=='clean':received=np.asarray(clean[ix],dtype=np.float32)
  else:
   cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
   received=[]
   for index in ix:
    pid=str(ids[index]);session='rx%d:day%d:eq1'%(int(rx[index]),int(day[index]));hw=receiver_for_session(cfg,2026110205,session)
    sample=np.asarray(clean[index],dtype=np.float64);signal,_=normalize_input(sample[0]+1j*sample[1])
    result,meta=ResidualChannel(cfg,stable_seed(2026110202,namespace,view,pid),hw).process(signal)
    assert np.isfinite(result).all() and not meta['channel_equalization_applied']
    received.append(np.stack((result.real,result.imag)).astype(np.float32))
   received=np.stack(received)
  obs={k:[] for k in ('probs','energy','rich')}
  for start in range(0,len(ix),32):
   x=core.tensor(received[start:start+32],'cpu');anchor=dura(x)
   feat=teacher.features(anchor);base=teacher.classify_features(feat);outputs=(x,anchor,boundary(x,anchor),feature(x,anchor,feat,base))
   probs=torch.stack([F.softmax(teacher.classify_features(teacher.features(y)),1) for y in outputs],1)
   energy=torch.stack((x.square().mean((1,2)).clamp_min(1e-8).log(),anchor.square().mean((1,2)).clamp_min(1e-8).log()),1)
   raw_embed=F.normalize(teacher.features(x),dim=1);anchor_embed=F.normalize(feat,dim=1)
   def spectrum(z):
    wave=torch.complex(z[:,0,:],z[:,1,:]);power=torch.fft.fft(wave,dim=-1).abs().square().reshape(len(z),16,16).mean(-1)
    return power/power.sum(1,keepdim=True).clamp_min(1e-8)
   rich=torch.cat((raw_embed,anchor_embed,spectrum(x),spectrum(x-anchor)),dim=1)
   obs['probs'].append(probs.cpu());obs['energy'].append(energy.cpu());obs['rich'].append(rich.cpu())
  views[view]={k:np.from_dlpack(torch.cat(v)) for k,v in obs.items()}
 print('SOURCE_EXPERT_OUTPUTS_FIXED',flush=True)
# Labels are joined only after all held-out expert outputs are fixed.
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[ix]=='L_s';assert np.all(labels[ix[~labeled]]==-1)
ref_probs=views['clean']['probs'][:,0,:];teacher_ref=ref_probs.argmax(1);conf=ref_probs.max(1)
target=np.where(labeled,labels[ix],teacher_ref);valid=labeled|(conf>=.95);results=[]
for view in view_names:
 values=views[view];features=np.concatenate((values['probs'].reshape(len(ix),24),values['energy'],values['rich']),1)
 scaled=np.clip((features-mean)/std,-10,10).astype(np.float32)
 with torch.no_grad():prob=np.from_dlpack(F.softmax(router(core.tensor(scaled,'cpu')),1).cpu().contiguous())
 alt=prob.copy();alt[:,1]=-1.;choice=alt.argmax(1);gap=alt[np.arange(len(ix)),choice]-prob[:,1]
 selected=np.where(gap>=threshold,choice,1);pred=values['probs'].argmax(2);output=pred[np.arange(len(ix)),selected];base=pred[:,1]
 old=base==target;new=output==target;lm=valid&labeled;ctold=base==teacher_ref;ctnew=output==teacher_ref
 row=dict(view=view,valid_count=int(valid.sum()),L_s_count=int(lm.sum()),U_confident_count=int((valid&~labeled).sum()),
  expert_choices={name:int((valid&(selected==j)).sum()) for j,name in enumerate(experts)},
  baseline=int((valid&old).sum()),router=int((valid&new).sum()),repairs=int((valid&~old&new).sum()),damages=int((valid&old&~new).sum()),
  L_s_baseline=int((lm&old).sum()),L_s_router=int((lm&new).sum()),clean_teacher_baseline=int(ctold.sum()),clean_teacher_router=int(ctnew.sum()))
 previous=next(x for x in expected['heldout'] if x['view']==view);assert row==previous,(row,previous)
 correct=pred==target[:,None];union=correct.any(1);three=correct[:,1:].any(1);base_correct=correct[:,1]
 oracle=dict(valid_count=int(valid.sum()),baseline=int((valid&base_correct).sum()),three_expert_oracle=int((valid&three).sum()),
  four_expert_oracle=int((valid&union).sum()),raw_unique=int((valid&~three&correct[:,0]).sum()),available_rescues=int((valid&~base_correct&union).sum()))
 results.append(dict(view=view,router=row,oracle=oracle));print('VERIFIED_SOURCE_VIEW',view,'oracle',json.dumps(oracle),flush=True)
print('ROUTER_REPLAY '+json.dumps(dict(status='VERIFIED_SOURCE_ROUTER_REPLAY',heldout_ids=820,models_refit=0,
 TX_labels_read_only_L_s=True,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,results=results,
 caveat='Source records were used to train fixed experts; mixed L_s true and confident U_s clean-teacher pseudo-targets. Oracle requires target truth and cannot be deployed.'),ensure_ascii=False),flush=True)
'''
remote=remote.replace('__INNER_SCRIPT__',repr(inner)).replace('__PAYLOAD__',repr(json.dumps(payload)))
proc=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=620)
stdout=proc.stdout.decode('utf8',errors='replace');print(stdout);print(proc.stderr.decode('utf8',errors='replace'))
if proc.returncode==0:
 line=next(s[len('ROUTER_REPLAY '):] for s in stdout.splitlines() if s.startswith('ROUTER_REPLAY '))
 path=Path(__file__).parent/'phase1_source_rich_physical_router_replay.json'
 if path.exists():
  assert json.loads(path.read_text(encoding='utf8')) == json.loads(line)
 else:
  with path.open('x',encoding='utf8') as f:f.write(line+'\n')
raise SystemExit(proc.returncode)
