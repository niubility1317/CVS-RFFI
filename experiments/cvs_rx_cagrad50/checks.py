import ast,copy,importlib,json,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d
from .objective import correction,solve,CLASS_W
from .observer import Observer
from .model import build

def real_gradient_check(m):
 g=torch.Generator().manual_seed(2026101541);x=torch.randn(10,2,256,generator=g);y=torch.arange(10)%6;rx=torch.tensor(d.RX).repeat_interleave(2);out=m.forward_identity_only(x);logits=out['tx_logits'];raw=F.cross_entropy(logits,y,label_smoothing=.01);pairs=[(n,p) for n,p in m.named_parameters() if n.startswith('id_backbone.') and n!=CLASS_W];params=[p for n,p in pairs];W=dict(m.named_parameters())[CLASS_W]
 def grad(loss):
  gg=torch.autograd.grad(loss,params+[W],retain_graph=True,allow_unused=True);return [torch.zeros_like(p).flatten() if v is None else v.flatten() for p,v in zip(params+[W],gg)]
 original=grad(raw);G=torch.stack([torch.cat(grad(F.cross_entropy(logits[rx==r],y[rx==r],label_smoothing=.01))[:-1]) for r in d.RX]);norms=[]
 for mode in ('NORM_CONTROL','RX_CAGRAD'):
  linear,info,_=correction(m,logits,y,rx,mode);actual=grad(raw+linear);want=G.mean(0)*info['norm_control_scale'] if mode=='NORM_CONTROL' else torch.tensor(info['coefficients'],dtype=G.dtype)@G
  assert torch.allclose(torch.cat(actual[:-1]),want,atol=1e-6,rtol=1e-5) and torch.allclose(actual[-1],original[-1],atol=1e-6,rtol=1e-5) and linear.item()==0
  norms.append(want.norm());assert np.isfinite(info['relative_gap']) and info['trust_relative']<=.50001
 assert torch.allclose(norms[0],norms[1],atol=1e-6,rtol=1e-5)
 return True

def checks():
 torch.set_num_threads(2);gen=np.random.default_rng(41);G=gen.normal(size=(5,53));b=np.array([.1,.2,.3,.15,.25]);s=solve(G@G.T,b);a=np.array(s['coefficients']);assert not s['fallback'] and np.linalg.norm((a-b)@G)<=.50001*np.linalg.norm(b@G) and ((G@G.T)@a).min()>=((G@G.T)@b).min()-1e-5
 assert np.isfinite(solve(np.zeros((1,1)),[1.])['norm_control_scale'])
 neg=[]
 for A,w,c in [(np.eye(2),[.5,.5],1.),(np.eye(2),[.2,.2],.5),(np.eye(2),[1.,0.],.5),(np.eye(2),[1.],.5),(np.full((2,2),np.nan),[.5,.5],.5),(-np.eye(2),[.5,.5],.5)]:neg.append(lambda A=A,w=w,c=c:solve(A,w,c))
 c=d.config(d.training_rows()[0])
 for key,value in [('coefficient',1.),('role','U_s'),('groups',[1,3])]:
  bad=copy.deepcopy(c);bad['receiver_gradient'][key]=value;neg.append(lambda bad=bad:d.validate(bad))
 for key,value in [('epochs',51),('source_config',{}),('classifier','refit')]:
  bad=copy.deepcopy(c);bad[key]=value;neg.append(lambda bad=bad:d.validate(bad))
 for f in neg:
  try:f()
  except ValueError:pass
  else:raise AssertionError('Forbiddencontext accepted')
 tree=ast.parse((d.ROOT/'experiments/adv3b02_xuc/code/SSDG/train_ssdg.py').read_text(encoding='utf8'));hook=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='_labeled_identity_loss_hook is not None');compiled=compile(ast.Module(body=[hook],type_ignores=[]),'<nativeRXgradienthook>','exec')
 initial=None;outputs=[];changes=[];g=torch.Generator().manual_seed(42);x=torch.randn(10,2,256,generator=g);y=torch.arange(10)%6;rx=torch.tensor(d.RX).repeat_interleave(2);args=SimpleNamespace(label_smoothing=.01,lambda_u=.16,lambda_ent=.01,ema_decay=.999)
 with tempfile.TemporaryDirectory() as tmp:
  for row in d.training_rows():
   c=d.config(row);na=d.make_args(c,'cpu');assert (na.epochs,na.label_epochs,na.pseudo_epochs)==(50,33,17) and na.use_unlabeled and na.use_ema_teacher and not na.freeze_backbone
   m=build(c,'cpu',design=d);state={k:v.clone() for k,v in m.state_dict().items()}
   if initial is None:initial=state;real_gradient_check(m)
   else:assert all(torch.equal(v,state[k]) for k,v in initial.items())
   p=Path(tmp)/row['arm'];p.mkdir();obs=Observer();obs.initialize(m,c,p);obs.steps=15;torch.manual_seed(43);m.train();out=m.forward_identity_only(x);raw=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
   ns=dict(_labeled_identity_loss_hook=obs.labeled,model=m,args=args,epoch=1,batch_idx=16,out_l={'tx_logits':out['tx_logits']},y_l=y,receiver_l_base=rx,labeled_clean_count=10,loss_tx_l=raw);exec(compiled,ns);selected=ns['loss_tx_l'];assert len(obs.grad)==1 and selected.item()==raw.item()
   u=out['id_feat_joint'].square().mean();assert obs.unlabeled(model=m,args=args,epoch=1,batch_idx=16,loss_l=selected,loss_u=u,loss_ent=u,active=False,logits=None,pseudo=None,mask=None) is u
   for role in ('U_s','V','query'):
    try:correction(m,out['tx_logits'],y,rx,row['arm'],role=role)
    except ValueError:pass
    else:raise AssertionError('NonL correction accepted')
   optim=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.0001);optim.zero_grad();selected.backward();named=dict(m.named_parameters());assert all(named[n].grad is not None and named[n].grad.norm()>0 for n in d.PARAMETERS) and named[CLASS_W].grad.norm()>0;torch.nn.utils.clip_grad_norm_(m.parameters(),5.);optim.step();changes.append([int((named[n]!=state[n]).sum()) for n in d.PARAMETERS]);assert all(changes[-1]) and int((named[CLASS_W]!=state[CLASS_W]).sum())>0
   m.eval()
   with torch.no_grad():z=m.forward_identity_only(x)['id_feat_joint'];saved=named[CLASS_W].clone();named[CLASS_W].zero_();same=m.forward_identity_only(x)['id_feat_joint'];named[CLASS_W].copy_(saved)
   assert torch.equal(z,same);outputs.append(z);obs.stream.close()
 assert not torch.equal(outputs[0],outputs[1])
 for name in ('source','predict','score','feature_check','audit','preflight','dispatch','publish'):importlib.import_module('experiments.cvs_rx_cagrad50.'+name)
 from .publish import configure,transport
 configure();compile(transport.REMOTE,'<remote>','exec');compile(transport.INSPECT,'<inspect>','exec');assert 'experiments.cvs_rx_cagrad50.preflight' in transport.REMOTE and 'experiments.cvs_rx_cagrad50.dispatch' in transport.REMOTE and 'experiments/cvs_rx_cagrad50' in transport.PATHS
 return dict(status='PASS',focused_negatives=len(neg)+6,actual_native_hook=True,full195041_gradient_formula=True,same_state_normmatched=True,default_W_gradient_original_only=True,original_U_returned_unchanged=True,identical_initial_states=True,synthetic_updates_per_arm=1,actual_body_changed=changes,actual_joint160_difference=True,oldW_zeroing_preserves_z=True,source_only=True,real_IQ_or_checkpoint_read=False,head_fit_updates=0)
if __name__=='__main__':print(json.dumps(checks()))
