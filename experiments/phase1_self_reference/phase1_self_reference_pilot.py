import pathlib,sys,torch,numpy as np,json
from torch.nn import functional as F
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet')
for n in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release','20261102-phase1-phase-shift-student-s2026092701-r02-release/experiments/phase1_phase_shift_restorer']:sys.path.insert(0,str(root/'releases'/n))
import phase_shift_source as s
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
"""Source-pilot input-only periodic self-reference IQ restorer."""
import torch
from torch import nn
class Block(nn.Module):
 def __init__(self,c,d):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(c,c,5,padding=2*d,dilation=d),nn.SiLU(),nn.Conv1d(c,c,1))
 def forward(self,x):return x+.2*self.net(x)
class SelfReferenceNet(nn.Module):
 def __init__(self,channels=64):
  super().__init__();self.input=nn.Conv1d(5,channels,9,padding=4);self.blocks=nn.Sequential(*(Block(channels,d) for d in (1,2,4,8,16,32)));self.output=nn.Conv1d(channels,4,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 @staticmethod
 def reference(z):
  a=z[:,:64];b=z[:,80:144];dot=(a*b.conj()).sum(1);rot=dot/dot.abs().clamp_min(1e-12);corr=dot.abs()/(a.abs().square().sum(1)*b.abs().square().sum(1)).clamp_min(1e-20).sqrt();ref=z.clone();ref[:,:64]=rot[:,None]*b;ref[:,80:144]=rot.conj()[:,None]*a;return ref,corr
 def forward(self,iq):
  if iq.ndim!=3 or iq.shape[1:]!=(2,256):raise ValueError('Expected [batch,2,256] IQ')
  rms=iq.square().mean((1,2),keepdim=True).clamp_min(1e-12).sqrt();x=iq/rms;z=torch.complex(x[:,0],x[:,1]);ref,corr=self.reference(z);diff=ref-z;features=torch.stack((z.real,z.imag,diff.real,diff.imag,corr[:,None].expand_as(z.real)),1);p=self.output(self.blocks(self.input(features)));gain=torch.complex(torch.tanh(p[:,0]),torch.tanh(p[:,1]));residual=torch.complex(torch.tanh(p[:,2]),torch.tanh(p[:,3]));out=z+gain*diff+.5*residual;return torch.stack((out.real,out.imag),1)*rms

torch.set_num_threads(2);torch.backends.mkldnn.enabled=True;torch.manual_seed(2026120154);smoke='--smoke' in sys.argv;nt=32 if smoke else 2048;nh=16 if smoke else 256;cache,clean,rx,training,_,_=s.source_contract(root);r1,sc1=s.epoch_data(cache,11);r2,sc2=s.epoch_data(cache,14);rng=np.random.default_rng(2026120155);pool=training[(sc1[training]!='clean')&(sc2[training]!='clean')];ids=rng.choice(pool,size=nt+nh,replace=False);train_ids=ids[:nt];held=ids[nt:];assert not np.intersect1d(train_ids,held).size
phase,gate=s.initial_models(root,'cpu');phase.eval().requires_grad_(False);teacher,_=core.teacher(root,'cpu');net=SelfReferenceNet();opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.001);losses=[];active_count=0
for st in range(0,len(train_ids),32):
 idx=train_ids[st:st+32];x=core.tensor(r1[idx],'cpu');y=core.tensor(clean[idx],'cpu')
 with torch.no_grad():active=gate(x).sigmoid()>.8
 if not active.any():continue
 x=x[active];y=y[active];active_count+=len(x)
 with torch.no_grad():a=phase.anchor(x);cf=teacher.features(y);cl=teacher.classify_features(cf);soft=F.softmax(cl/2,1);aligned,_=s.align(y,a)
 out=net(a);feat=teacher.features(out);logits=teacher.classify_features(feat);kl=F.kl_div(F.log_softmax(logits/2,1),soft,reduction='none').sum(1).mean()*4;finger=(1-F.cosine_similarity(feat,cf,dim=1,eps=1e-8)).mean();den=aligned.square().mean((1,2)).clamp_min(1e-8);iq=((out-aligned).square().mean((1,2))/den).mean();inv=s.complex_gain_invariant_error(out,aligned).mean();loss=kl+.5*finger+.3*iq+.2*inv
 if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source loss')
 opt.zero_grad(set_to_none=True)
 if smoke:
  zero=torch.autograd.grad(loss*0,list(net.parameters()),retain_graph=True,allow_unused=True)
  if any(g is not None and (not torch.isfinite(g).all() or torch.count_nonzero(g)!=0) for g in zero):raise FloatingPointError('Nonzero zero-gradient through ground F')
 loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
 if net.output.weight.grad is None or not torch.isfinite(net.output.weight.grad).all() or not net.output.weight.grad.abs().sum()>0:raise FloatingPointError('Missing finite self-reference gradient')
 opt.step();losses.append(float(loss.detach()))
print('SELF_REFERENCE_TRAINED',len(losses),'active',active_count,'smoke',smoke,flush=True)
net.eval().requires_grad_(False);d=prior.frozen_dura(root,'cpu');rows=[]
with torch.no_grad():
 for st in range(0,len(held),16):
  idx=held[st:st+16];x=core.tensor(r2[idx],'cpu');y=core.tensor(clean[idx],'cpu');a=phase.anchor(x);z=torch.complex(a[:,0],a[:,1]);ref,corr=SelfReferenceNet.reference(z);mix=z+.1*(ref-z);mix=torch.stack((mix.real,mix.imag),1);on=gate(x).sigmoid()>.8;candidate=net(a);p=teacher.classify_features(teacher.features(y)).softmax(1);good=p.max(1).values>=.95;target=p.argmax(1);baseline=d(x);waves=[x,baseline,torch.where(on[:,None,None],a,x),torch.where((on&(corr>=.8))[:,None,None],mix,torch.where(on[:,None,None],a,x)),torch.where(on[:,None,None],candidate,x)];den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);cols=[good.float()]
  for w in waves:
   pred=teacher.classify_features(teacher.features(w)).argmax(1);_,e=s.align(w,y);cols.extend(((pred==target).float(),e/den))
  rows.append(torch.stack(cols,1))
v=np.from_dlpack(torch.cat(rows).contiguous());result={};names=['raw','DURA','anchor','fixed_repeat','self_reference_net']
for view in s.VIEWS[1:]:
 a=v[sc2[held]==view];g=a[a[:,0]>.5];result[view]={'count':len(a),'valid':len(g),'methods':{n:{'correct':int(g[:,1+2*i].sum()),'aligned_nmse':float(a[:,2+2*i].mean()) if len(a) else None} for i,n in enumerate(names)}}
g=v[v[:,0]>.5];result['all']={'count':len(v),'valid':len(g),'methods':{n:{'correct':int(g[:,1+2*i].sum()),'aligned_nmse':float(v[:,2+2*i].mean())} for i,n in enumerate(names)}}
print('SELF_REFERENCE_RESULT',json.dumps({'smoke':smoke,'source_only':True,'train_ids':len(train_ids),'heldout_from_train':len(held),'training_cache':11,'evaluation_cache':14,'updates':len(losses),'active_records':active_count,'loss_mean':float(np.mean(losses)),'F_frozen':True,'phase_prior':'source-trained previously with DURA, DURA not used in inference','DURA_loaded_after_training':True,'TX_labels_read':0,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'target':'high-confidence F(clean) pseudo label, not true accuracy','limitations':'small source pilot; reused source identities; no checkpoint saved, not Phase1 test','results':result},allow_nan=False),flush=True)
