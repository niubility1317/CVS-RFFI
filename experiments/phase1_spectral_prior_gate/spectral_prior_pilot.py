import pathlib,sys,torch,numpy as np,json
from torch.nn import functional as F
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet')
for n in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release','20261102-phase1-phase-shift-student-s2026092701-r02-release/experiments/phase1_phase_shift_restorer']:sys.path.insert(0,str(root/'releases'/n))
import phase_shift_source as s
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
"""Source-pilot spectral correction to a label-free waveform prior."""
import torch
from torch import nn
class Block(nn.Module):
 def __init__(self,c,d):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(c,c,5,padding=2*d,dilation=d,padding_mode='circular'),nn.SiLU(),nn.Conv1d(c,c,1))
 def forward(self,x):return x+.2*self.net(x)
class SpectralPriorGate(nn.Module):
 def __init__(self,channels=48):
  super().__init__();self.input=nn.Conv1d(6,channels,5,padding=2,padding_mode='circular');self.blocks=nn.Sequential(*(Block(channels,d) for d in (1,2,4,8)));self.output=nn.Conv1d(channels,2,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,iq,prior):
  if iq.shape!=prior.shape or iq.ndim!=3 or iq.shape[1:]!=(2,256):raise ValueError('matching [batch,2,256] IQ required')
  rms=iq.square().mean((1,2),keepdim=True).clamp_min(1e-12).sqrt();x=iq/rms;t=prior/rms;z=torch.complex(x[:,0],x[:,1]);v=torch.complex(t[:,0],t[:,1]);X=torch.fft.fft(z,norm='ortho');T=torch.fft.fft(v,norm='ortho');D=T-X;features=torch.stack((X.real,X.imag,T.real,T.imag,D.real,D.imag),1);p=self.output(self.blocks(self.input(features)));gain=torch.complex(torch.tanh(p[:,0]),torch.tanh(p[:,1]));y=torch.fft.ifft(X+gain*D,norm='ortho');return torch.stack((y.real,y.imag),1)*rms

torch.set_num_threads(2);torch.backends.mkldnn.enabled=True;torch.manual_seed(2026112454);smoke='--smoke' in sys.argv;nt=32 if smoke else 2048;nh=16 if smoke else 256;cache,clean,rx,training,_,_=s.source_contract(root);r1,sc1=s.epoch_data(cache,11);r2,sc2=s.epoch_data(cache,14);rng=np.random.default_rng(2026112455);bank_ids=rng.choice(training,size=8192,replace=False);pool=np.setdiff1d(training,bank_ids,assume_unique=True);pool=pool[(sc1[pool]!='clean')&(sc2[pool]!='clean')];ids=rng.choice(pool,size=nt+nh,replace=False);train_ids=ids[:nt];held_ids=ids[nt:];assert len(np.intersect1d(bank_ids,ids))==0 and len(np.intersect1d(train_ids,held_ids))==0
m,gate=s.initial_models(root,'cpu');m.eval();teacher,_=core.teacher(root,'cpu');assert all(not p.requires_grad for p in teacher.parameters());bank=[]
with torch.no_grad():
 for st in range(0,len(bank_ids),256):
  y=core.tensor(clean[bank_ids[st:st+256]],'cpu');bank.append(torch.complex(y[:,0],y[:,1]))
 bank=torch.cat(bank);bc=bank[:,64:-64];power=bc.abs().square().sum(1).clamp_min(1e-12);unit=bc/power.sqrt()[:,None]
def prepare(ids,received):
 aa=[];tt=[];xx=[];yy=[];gg=[]
 with torch.no_grad():
  for st in range(0,len(ids),32):
   j=ids[st:st+32];x=core.tensor(received[j],'cpu');y=core.tensor(clean[j],'cpu');a=m.anchor(x);az=torch.complex(a[:,0],a[:,1]);ac=az[:,64:-64];score=(ac@unit.conj().T).abs()/ac.abs().square().sum(1).clamp_min(1e-12).sqrt()[:,None];vals,ind=score.topk(16,dim=1);t=bank[ind];gain=(t[:,:,64:-64].conj()*ac[:,None,:]).sum(-1)/power[ind];weight=F.softmax((vals-vals[:,:1])/.05,dim=1);prior=(t*gain[:,:,None]*weight[:,:,None]).sum(1);aa.append(a);tt.append(torch.stack((prior.real,prior.imag),1));xx.append(x);yy.append(y);gg.append(gate(x).sigmoid()>.8)
 return tuple(torch.cat(v,0) for v in (aa,tt,xx,yy,gg))
A,T,X,Y,G=prepare(train_ids,r1);AH,TH,XH,YH,GH=prepare(held_ids,r2);net=SpectralPriorGate();opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.001);losses=[];active_count=0
for st in range(0,len(A),32):
 a=A[st:st+32];t=T[st:st+32];y=Y[st:st+32];active=G[st:st+32]
 if not active.any():continue
 a=a[active];t=t[active];y=y[active];active_count+=len(a)
 with torch.no_grad():cf=teacher.features(y);cl=teacher.classify_features(cf);soft=F.softmax(cl/2,1);aligned,_=s.align(y,a)
 out=net(a,t);feat=teacher.features(out);logit=teacher.classify_features(feat);kl=F.kl_div(F.log_softmax(logit/2,1),soft,reduction='none').sum(1).mean()*4;finger=(1-F.cosine_similarity(feat,cf,dim=1,eps=1e-8)).mean();den=aligned.square().mean((1,2)).clamp_min(1e-8);iq=((out-aligned).square().mean((1,2))/den).mean();inv=s.complex_gain_invariant_error(out,aligned).mean();loss=kl+.5*finger+.3*iq+.2*inv
 if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source loss')
 opt.zero_grad(set_to_none=True)
 if smoke:
  z=torch.autograd.grad(loss*0,list(net.parameters()),retain_graph=True,allow_unused=True)
  if any(q is not None and (not torch.isfinite(q).all() or torch.count_nonzero(q)!=0) for q in z):raise FloatingPointError('Nonzero zero-gradient through source ground F')
 loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
 if net.output.weight.grad is None or not torch.isfinite(net.output.weight.grad).all() or not net.output.weight.grad.abs().sum()>0:raise FloatingPointError('Missing spectral gate gradient')
 opt.step();losses.append(float(loss.detach()))
print('SPECTRAL_PRIOR_TRAINED',len(losses),'active',active_count,'smoke',smoke,flush=True)
net.eval().requires_grad_(False);d=prior.frozen_dura(root,'cpu');rows=[]
with torch.no_grad():
 for st in range(0,len(AH),16):
  a=AH[st:st+16];t=TH[st:st+16];x=XH[st:st+16];y=YH[st:st+16];on=GH[st:st+16];p=teacher.classify_features(teacher.features(y)).softmax(1);good=p.max(1).values>=.95;target=p.argmax(1);baseline=d(x);waves=[x,baseline,torch.where(on[:,None,None],a,x),torch.where(on[:,None,None],a+.05*(t-a),x),torch.where(on[:,None,None],a+.25*(t-a),x),torch.where(on[:,None,None],net(a,t),x)];den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12);cols=[good.float()]
  for w in waves:
   pred=teacher.classify_features(teacher.features(w)).argmax(1);_,e=s.align(w,y);cols.extend(((pred==target).float(),e/den))
  rows.append(torch.stack(cols,1))
v=np.from_dlpack(torch.cat(rows).contiguous());res={};names=['raw','DURA','anchor','blend05','blend25','spectral_gate']
for view in s.VIEWS[1:]:
 a=v[sc2[held_ids]==view];good=a[:,0]>.5;g=a[good]
 res[view]={'count':len(a),'valid':len(g),'methods':{n:{'correct':int(g[:,1+2*i].sum()),'aligned_nmse':float(a[:,2+2*i].mean()) if len(a) else None} for i,n in enumerate(names)}}
g=v[v[:,0]>.5];res['all']={'count':len(v),'valid':len(g),'methods':{n:{'correct':int(g[:,1+2*i].sum()),'aligned_nmse':float(v[:,2+2*i].mean())} for i,n in enumerate(names)}}
print('SPECTRAL_PRIOR_RESULT',json.dumps({'smoke':smoke,'source_only':True,'train_ids':len(train_ids),'heldout_from_train':len(held_ids),'dictionary_ids':len(bank_ids),'neighbors':16,'training_cache':11,'evaluation_cache':14,'updates':len(losses),'active_records':active_count,'loss_mean':float(np.mean(losses)),'F_frozen':True,'phase_prior':'previously source-trained with DURA, no DURA inference','DURA_loaded_after_training':True,'TX_labels_read':0,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'target':'high-confidence F(clean) pseudo label, not true accuracy','limitations':'small exploratory IQ-restoration pilot; disjoint physical-ID dictionary can introduce other transmitters fingerprints; no checkpoint saved, not Phase1 test','results':res},allow_nan=False),flush=True)
