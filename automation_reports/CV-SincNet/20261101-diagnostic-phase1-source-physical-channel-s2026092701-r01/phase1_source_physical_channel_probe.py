"""Fresh source IDs, independently seeded Phase1 practical-channel stress."""
import subprocess
from pathlib import Path

remote=r'''
from pathlib import Path
import os,subprocess
r=Path('/home/szu2070436088/2510044040/CV-SincNet');run='20261101-phase1-noise-guard-router-s2026092701-r01'
release=r/'releases'/(run+'-release-v2');native=r/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='a055a0c556a1a0dda1c34aadfe7c6eb86c56a353'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',__INNER_SCRIPT__],cwd=release,env=env,check=True,timeout=600)
'''
inner=r'''
from pathlib import Path
import hashlib,json,numpy as np,torch
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,guarded_router_experiment as g,feature_conditioned_experiment as fs
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
models=g.load_models(root,'cpu');teacher,contract,dura,boundary,feature,_,_,_=models
cache,clean,roles=core.data(root,contract)
from leo_practical.channel import Config,receiver_for_session,stable_seed,normalize_input
from leo_practical.residual import ResidualChannel
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);rx=np.load(cache/'train/rx.npy',allow_pickle=False);day=np.load(cache/'train/day.npy',allow_pickle=False)
assert len(ids)==len(rx)==len(day)==len(roles)==63000
groups=[]
for role,count,initial,n in (('L_s',6300,128,1024),('U_s',56700,384,3072)):
 available=np.flatnonzero(roles==role);prior=available[np.linspace(0,count-1,initial,dtype=int)];available=np.setdiff1d(available,prior,assume_unique=True)
 for _ in range(3):
  old=available[np.linspace(0,len(available)-1,n,dtype=int)];available=np.setdiff1d(available,old,assume_unique=True)
 take=128 if role=='L_s' else 384;groups.append(available[np.linspace(0,len(available)-1,take,dtype=int)])
ix=np.concatenate(groups);assert len(ix)==len(np.unique(ix))==512
views={};namespace='phase1-source-physical-probe-2026110181'
scenes=('practical_low_suburban','practical_mid_urban','practical_low_urban')
with torch.no_grad():
 for scene in ('clean',)+scenes:
  if scene=='clean':received=np.asarray(clean[ix],dtype=np.float32)
  else:
   cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
   received=[];sessions={}
   for index in ix:
    pid=str(ids[index]);session='rx%d:day%d:eq1'%(int(rx[index]),int(day[index]));hw=receiver_for_session(cfg,2026110182,session)
    if session not in sessions:sessions[session]=hw
    sample=np.asarray(clean[index],dtype=np.float64);signal,_=normalize_input(sample[0]+1j*sample[1])
    result,meta=ResidualChannel(cfg,stable_seed(2026110181,namespace,scene,pid),hw).process(signal)
    assert np.isfinite(result).all() and not meta['channel_equalization_applied']
    received.append(np.stack((result.real,result.imag)).astype(np.float32))
   received=np.stack(received)
  obs={k:[] for k in ('probs','energy','choice')}
  for start in range(0,len(ix),32):
   x=core.tensor(received[start:start+32],'cpu');anchor=dura(x);feat=teacher.features(anchor);baseline=teacher.classify_features(feat)
   outputs=(x,anchor,boundary(x,anchor),feature(x,anchor,feat,baseline))
   probs=torch.stack([F.softmax(teacher.classify_features(teacher.features(y)),1) for y in outputs],1)
   energy=torch.stack((x.square().mean((1,2)).clamp_min(1e-8).log(),anchor.square().mean((1,2)).clamp_min(1e-8).log()),1)
   _,_,choice=g.route(models,x);obs['probs'].append(probs.cpu());obs['energy'].append(energy.cpu());obs['choice'].append(choice.cpu())
  views[scene]={k:np.from_dlpack(torch.cat(v)) for k,v in obs.items()};print('SOURCE_PHYSICAL_OUTPUT_FIXED',scene,flush=True)
print('ALL_SOURCE_PHYSICAL_OUTPUTS_FIXED',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[ix]=='L_s';assert np.all(labels[ix[~labeled]]==-1)
ref=views['clean']['probs'][:,0,:];target=np.where(labeled,labels[ix],ref.argmax(1));valid=labeled|(ref.max(1)>=.95);results=[]
for scene,v in views.items():
 pred=v['probs'].argmax(2);correct=pred==target[:,None];choice=v['choice'];out=pred[np.arange(len(ix)),choice];old=correct[:,1];new=out==target
 three=correct[:,1:].any(1);four=correct.any(1);q=np.quantile(v['energy'][:,1],[.05,.5,.95])
 row=dict(scene=scene,valid=int(valid.sum()),labeled=int((valid&labeled).sum()),pseudo=int((valid&~labeled).sum()),
  raw=int((valid&correct[:,0]).sum()),dura=int((valid&old).sum()),router=int((valid&new).sum()),
  repairs=int((valid&~old&new).sum()),damages=int((valid&old&~new).sum()),
  oracle_three=int((valid&three).sum()),oracle_four=int((valid&four).sum()),raw_unique=int((valid&~three&correct[:,0]).sum()),
  choices=[int((valid&(choice==i)).sum()) for i in range(4)],dura_log_energy_quantiles=[float(z) for z in q])
 results.append(row)
print('SOURCE_PHYSICAL_PROBE '+json.dumps(dict(status='SOURCE_ONLY_PHYSICAL_SIMULATOR_PROBE',source_ids=512,results=results,
 simulator='LEO practical residual, source independent seed and sessions; fs 25 MHz fc 2.462 GHz, reference SNR 30 dB',
 noise_seed=2026110181,receiver_seed=2026110182,source_only=True,source_expert_training_seen_ids=True,
 U_s_labels_read=0,V_reads=0,test_IQ_reads=0,expert_decisions_before_source_labels=True,
 caveat='Simulation is not validated real satellite data. Mixed actual L_s labels and high-confidence U_s clean-teacher pseudo-targets. Oracle needs targets and cannot be deployed.'),ensure_ascii=False),flush=True)
'''
remote=remote.replace('__INNER_SCRIPT__',repr(inner))
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=620)
stdout=p.stdout.decode('utf8',errors='replace');print(stdout);print(p.stderr.decode('utf8',errors='replace'))
if p.returncode==0:
 line=next(s[len('SOURCE_PHYSICAL_PROBE '):] for s in stdout.splitlines() if s.startswith('SOURCE_PHYSICAL_PROBE '))
 with (Path(__file__).parent/'phase1_source_physical_channel_probe.json').open('x',encoding='utf8') as f:f.write(line+'\n')
raise SystemExit(p.returncode)
