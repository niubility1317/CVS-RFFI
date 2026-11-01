"""Small source-only conditional diffusion restoration pilot."""
import subprocess
from pathlib import Path

inner=r'''
from pathlib import Path
import json,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,physical_rich_router_experiment as g,feature_conditioned_experiment as fs
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
models=g.load_models(root,'cpu');teacher,contract,dura,*_=models;cache,clean,roles=core.data(root,contract)
from leo_practical.channel import Config,receiver_for_session,stable_seed,normalize_input
from leo_practical.residual import ResidualChannel
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);rx=np.load(cache/'train/rx.npy',allow_pickle=False);day=np.load(cache/'train/day.npy',allow_pickle=False)
rng=np.random.default_rng(2026110301);groups=[]
for role,take in (('L_s',1024),('U_s',3072)):
 available=np.flatnonzero(roles==role);groups.append(np.sort(rng.choice(available,size=take,replace=False)))
fixed=np.concatenate(groups);assert len(fixed)==len(np.unique(fixed))==4096
source_rx=rx[fixed];source_day=day[fixed];split=np.full(len(fixed),-1,dtype=np.int64)
split[np.isin(source_rx,[1,3,4])&np.isin(source_day,[1,2])]=0
split[np.isin(source_rx,[4,6])&(source_day==3)]=1
split[(source_rx==8)&(source_day==3)]=2
assert all(int((split==i).sum())>200 for i in range(3))
views=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban');namespace='phase1-source-conditional-diffusion-2026110302';data={}
with torch.no_grad():
 for view in views:
  if view=='clean':received=np.asarray(clean[fixed],dtype=np.float32)
  else:
   cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
   received=[]
   for j in fixed:
    pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110303,session)
    v=np.asarray(clean[j],dtype=np.float64);signal,_=normalize_input(v[0]+1j*v[1]);result,meta=ResidualChannel(cfg,stable_seed(2026110302,namespace,view,pid),hw).process(signal)
    assert np.isfinite(result).all() and not meta['channel_equalization_applied'];received.append(np.stack((result.real,result.imag)).astype(np.float32))
   received=np.stack(received)
  anchors=[];rawp=[];basep=[]
  for start in range(0,len(fixed),32):
   x=core.tensor(received[start:start+32],'cpu');a=dura(x);anchors.append(a.cpu())
   rawp.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu())
   basep.append(F.softmax(teacher.classify_features(teacher.features(a)),1).cpu())
  data[view]=dict(received=core.tensor(received,'cpu'),anchor=torch.cat(anchors),rawp=np.from_dlpack(torch.cat(rawp).contiguous()),basep=np.from_dlpack(torch.cat(basep).contiguous()))
  print('DIFFUSION_SOURCE_BASELINES_FIXED',view,flush=True)
print('ALL_SOURCE_BASELINES_FIXED_BEFORE_SOURCE_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1)
ref=data['clean']['rawp'];teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target)
n=len(fixed);reference=core.tensor(np.asarray(clean[fixed],dtype=np.float32),'cpu')
received=torch.cat([data[v]['received'] for v in views]);anchors=torch.cat([data[v]['anchor'] for v in views]);clean_all=reference.repeat(len(views),1,1)
train_ix=np.flatnonzero(np.tile(split,len(views))==0)
class Block(nn.Module):
 def __init__(self,dilation):
  super().__init__();self.a=nn.Conv1d(32,32,3,padding=dilation,dilation=dilation);self.b=nn.Conv1d(32,32,3,padding=1)
 def forward(self,x):return x+.2*self.b(F.gelu(self.a(F.gelu(x))))
class NoiseNet(nn.Module):
 def __init__(self):
  super().__init__();self.enter=nn.Conv1d(7,32,5,padding=2);self.blocks=nn.Sequential(Block(1),Block(2),Block(4));self.exit=nn.Conv1d(32,2,3,padding=1)
 def forward(self,z,a,x,sigma):
  t=sigma[:,None,None].expand(-1,1,z.shape[-1]);return self.exit(self.blocks(self.enter(torch.cat((z,a,x,t),1))))
torch.manual_seed(2026110304);net=NoiseNet();opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.0001);rng=np.random.default_rng(2026110305)
sigmas=torch.tensor([.05,.10,.20,.30],dtype=torch.float32);alphas=(.1,.25,.5,1.);validation=[];best={'kind':'dura','validation_net':0};best_state=None
def generate(z,a,x):
 for s,next_s in ((.30,.20),(.20,.10),(.10,.05),(.05,0.)):
  st=torch.full((len(z),),s);noise=net(z,a,x,st);clean_hat=(z-s*noise)/(1-s*s)**.5
  z=(1-next_s*next_s)**.5*clean_hat+next_s*noise
 return z
evaluation_cache={}
def evaluate(alpha,part,group=None):
 selection=np.flatnonzero(split==part);group_mask=np.ones(len(selection),dtype=bool) if group is None else source_rx[selection]==group
 per={};clean_net=0
 with torch.no_grad():
  for view in views:
   baseline=data[view]['basep'].argmax(1);gkey=('generated',part,view);okey=('output',part,view,alpha)
   if gkey not in evaluation_cache:
    generated=[]
    for start in range(0,len(selection),64):
     idx=torch.tensor(selection[start:start+64].tolist(),dtype=torch.long);a=data[view]['anchor'][idx];x=data[view]['received'][idx]
     generated.append(generate((1-.30**2)**.5*a,a,x).cpu())
    evaluation_cache[gkey]=torch.cat(generated)
   if okey not in evaluation_cache:
    outputs=[];generated=evaluation_cache[gkey]
    for start in range(0,len(selection),64):
     idx=torch.tensor(selection[start:start+64].tolist(),dtype=torch.long);a=data[view]['anchor'][idx];wave=a+alpha*(generated[start:start+len(idx)]-a)
     assert torch.isfinite(wave).all();outputs.append(teacher.classify_features(teacher.features(wave)).argmax(1).cpu())
    evaluation_cache[okey]=torch.cat(outputs).contiguous()
   output=np.from_dlpack(evaluation_cache[okey]);vi=valid[selection]&group_mask;lb=labeled[selection];truth=target[selection]
   old=baseline[selection]==truth;new=output==truth;lm=vi&lb
   per[view]=dict(valid_count=int(vi.sum()),L_s_count=int(lm.sum()),U_confident_count=int((vi&~lb).sum()),baseline=int((vi&old).sum()),student=int((vi&new).sum()),repairs=int((vi&~old&new).sum()),damages=int((vi&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()))
   if view=='clean':clean_net=int((group_mask&(output==teacher_target[selection])).sum()-(group_mask&(baseline[selection]==teacher_target[selection])).sum())
 return per,clean_net
for epoch in range(1,13):
 for start in range(0,len(train_ix),64):
  if start==0:order=rng.permutation(train_ix)
  ix=torch.tensor(order[start:start+64].tolist(),dtype=torch.long);s=sigmas[torch.randint(0,len(sigmas),(len(ix),))];eps=torch.randn_like(clean_all[ix]);z=(1-s[:,None,None]**2).sqrt()*clean_all[ix]+s[:,None,None]*eps
  pred=net(z,anchors[ix],received[ix],s);loss=F.mse_loss(pred,eps)
  if not torch.isfinite(loss):raise FloatingPointError('Finite diffusion noise loss required')
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
 if epoch in (2,5,8,12):
  net.eval();evaluation_cache.clear()
  for alpha in alphas:
   per,clean_net=evaluate(alpha,1);changes={v:per[v]['student']-per[v]['baseline'] for v in views[1:]};net_gain=sum(changes.values());groups_v={}
   for group in (4,6):
    gp,gclean=evaluate(alpha,1,group);count=int(((split==1)&(source_rx==group)).sum());gc={v:gp[v]['student']-gp[v]['baseline'] for v in views[1:]}
    groups_v[str(group)]=dict(n=count,net=sum(gc.values()),per_view=gc,clean_teacher_net=gclean)
   validation.append(dict(epoch=epoch,alpha=alpha,net=net_gain,per_view=changes,clean_teacher_net=clean_net,groups=groups_v))
   guards=all(z['net']>=0 and z['clean_teacher_net']>=-.01*z['n'] and min(z['per_view'].values())>=-.005*z['n'] for z in groups_v.values())
   if net_gain>best['validation_net'] and guards:
    best=dict(kind='conditional_diffusion',epoch=epoch,alpha=alpha,validation_net=net_gain);best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
  print('DIFFUSION_SOURCE_VALIDATION_COMPLETE',epoch,flush=True);net.train()
if best_state is None:heldout=None;clean_net=None;state=None
else:
 net.load_state_dict(best_state);net.eval();evaluation_cache.clear();heldout,clean_net=evaluate(best['alpha'],2);state={k:v.detach().tolist() for k,v in net.state_dict().items()}
print('SOURCE_CONDITIONAL_DIFFUSION_PROBE '+json.dumps(dict(status='SOURCE_ONLY_CONDITIONAL_DIFFUSION_PROBE',source_ids=n,train_ids=int((split==0).sum()),validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),excluded_ids=int((split<0).sum()),train_receivers=[1,3,4],validation_receivers=[4,6],heldout_receivers=[8],train_days=[1,2],validation_days=[3],heldout_days=[3],validation=validation,selected=best,heldout=heldout,heldout_clean_teacher_net=clean_net,model_state=state,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,baseline_outputs_before_source_labels=True,
 caveat='Conditioned deterministic four-step DDIM-like pilot, initialized from DURA; generated samples were not independently calibrated as physical satellite IQ. Source model split holds out receiver/day but frozen experts saw source records, and IDs may overlap previous diagnostics. Earlier Phase1 test findings motivated this exploration.'),ensure_ascii=False),flush=True)
'''
remote=r'''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',__INNER__],cwd=release,env=env,check=True,timeout=3600)
'''.replace('__INNER__',repr(inner))
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=3620)
stdout=p.stdout.decode('utf8',errors='replace');stderr=p.stderr.decode('utf8',errors='replace')
for line in stdout.splitlines():
 if not line.startswith('SOURCE_CONDITIONAL_DIFFUSION_PROBE '):print(line)
print(stderr)
if p.returncode==0:
 line=next(x[len('SOURCE_CONDITIONAL_DIFFUSION_PROBE '):] for x in stdout.splitlines() if x.startswith('SOURCE_CONDITIONAL_DIFFUSION_PROBE '))
 dest=Path(__file__).parent/'phase1_source_conditional_diffusion_probe.json'
 with dest.open('x',encoding='utf8') as f:f.write(line+'\n')
 print('SAVED_SOURCE_CONDITIONAL_DIFFUSION_RESULT',dest)
raise SystemExit(p.returncode)
