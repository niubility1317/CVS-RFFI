import ast,copy,importlib,json,tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from . import design as d
from .objective import objective,risks
from .observer import Observer

def checks():
 torch.set_num_threads(2);g=torch.Generator().manual_seed(31547);a=torch.randn(11,6,generator=g,dtype=torch.float64,requires_grad=True);y=torch.arange(11)%6;rx=torch.tensor([1]*3+[3]*2+[4]*2+[6]*2+[8]*2)
 chosen,balanced,var,counts,means=objective(a,y,rx,1.)
 manual=torch.stack([F.cross_entropy(a[rx==r],y[rx==r],label_smoothing=.01) for r in d.RX]);assert torch.allclose(means,manual,atol=1e-14) and torch.allclose(var,manual.var(unbiased=False),atol=1e-14) and torch.equal(chosen,balanced+var)
 z,b,v,_,_=objective(a,y,rx,0.);assert z is b and torch.equal(torch.autograd.grad(z,a,retain_graph=True)[0],torch.autograd.grad(b,a,retain_graph=True)[0])
 left,lb,lv,_,_=objective(a[:3],y[:3],rx[:3],1.);assert torch.allclose(left,F.cross_entropy(a[:3],y[:3],label_smoothing=.01),atol=1e-14) and lv==0
 neg=[]
 for role in ('U_s','V','query'):neg.append(lambda role=role:objective(a,y,rx,1.,role=role))
 neg.extend([lambda:objective(a,y,rx,.1),lambda:objective(a[:,:5],y,rx,1.),lambda:objective(a,y,rx*0,1.),lambda:objective(a,y+6,rx,1.),lambda:objective(a,y.float(),rx,1.),lambda:objective(torch.full_like(a,float('nan')),y,rx,1.)])
 for key,value in [('groups',[1,3]),('coefficient',10.),('role','U_s'),('mode','DRO')]:
  c=d.config(d.training_rows()[1]);c['receiver_risk'][key]=value;neg.append(lambda c=c:d.validate(c))
 for key,value in [('epochs',51),('source_config',{}),('classifier','linear_refit')]:
  c=d.config(d.training_rows()[0]);c[key]=value;neg.append(lambda c=c:d.validate(c))
 for f in neg:
  try:f()
  except ValueError:pass
  else:raise AssertionError('ForbiddenVREXcontextaccepted')
 tree=ast.parse((d.ROOT/'experiments/adv3b02_xuc/code/SSDG/train_ssdg.py').read_text(encoding='utf8'));hook=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='_labeled_identity_loss_hook is not None');compiled=compile(ast.Module(body=[hook],type_ignores=[]),'<actualnativeVREXhook>','exec')
 from .model import build
 initial=None;outputs=[];body_updates=[];args=SimpleNamespace(label_smoothing=.01,lambda_u=.16,lambda_ent=.01,ema_decay=.999);x=torch.randn(10,2,256,generator=g);labels=torch.arange(10)%6;receivers=torch.tensor(d.RX).repeat_interleave(2)
 with tempfile.TemporaryDirectory() as tmp:
  for row in d.training_rows():
   c=d.config(row);native_args=d.make_args(c,'cpu');assert (native_args.epochs,native_args.label_epochs,native_args.pseudo_epochs,native_args.weight_decay)==(50,33,17,.0001) and native_args.use_unlabeled and native_args.use_ema_teacher and not native_args.freeze_backbone
   m=build(c,'cpu',design=d);state={k:v.clone() for k,v in m.state_dict().items()}
   if initial is None:initial=state
   else:assert set(initial)==set(state) and all(torch.equal(v,state[k]) for k,v in initial.items())
   path=Path(tmp)/row['arm'];path.mkdir();obs=Observer();obs.initialize(m,c,path);obs.steps=15;m.train();torch.manual_seed(31548);out=m.forward_identity_only(x);raw=F.cross_entropy(out['tx_logits'],labels,label_smoothing=.01)
   ns=dict(_labeled_identity_loss_hook=obs.labeled,model=m,args=args,epoch=1,batch_idx=16,out_l={'tx_logits':out['tx_logits']},y_l=labels,receiver_l_base=receivers,labeled_clean_count=len(labels),loss_tx_l=raw);exec(compiled,ns);selected=ns['loss_tx_l'];assert obs.steps==16 and len(obs.grad)==1
   vct=obs.grad[0]['vectors'];alpha=c['receiver_risk']['coefficient']
   assert all(torch.allclose(s,b+alpha*v,atol=1e-6,rtol=1e-5) for s,b,v in zip(vct['selected'],vct['balanced'],vct['variance']))
   zero=out['tx_logits'].sum()*0;assert obs.unlabeled(model=m,args=args,epoch=1,batch_idx=16,loss_l=selected,loss_u=zero,loss_ent=zero,active=False,logits=None,pseudo=None,mask=None) is zero
   obs.steps=33*222+1;assert obs.unlabeled(model=m,args=args,epoch=34,batch_idx=1,loss_l=selected,loss_u=zero,loss_ent=zero,active=True,logits=out['tx_logits'],pseudo=labels,mask=torch.ones(10,dtype=torch.bool)) is zero and obs.U_packets==10
   optimizer=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.0001);optimizer.zero_grad();selected.backward();named=dict(m.named_parameters());W=m.id_backbone.encoder.core.id_backbone.cls_head.weight
   assert W.shape==(6,160) and W.grad is not None and W.grad.norm()>0 and all(named[n].grad is not None and named[n].grad.norm()>0 for n in d.PARAMETERS)
   torch.nn.utils.clip_grad_norm_(m.parameters(),5.);optimizer.step();changes=[int((named[n].detach()!=state[n]).sum()) for n in d.PARAMETERS];assert all(changes) and int((W.detach()!=state['id_backbone.encoder.core.id_backbone.cls_head.weight']).sum())>0;body_updates.append(changes)
   m.eval()
   with torch.no_grad():z=m.forward_identity_only(x)['id_feat_joint'];saved=W.clone();W.zero_();same=m.forward_identity_only(x)['id_feat_joint'];W.copy_(saved)
   assert torch.equal(z,same);outputs.append(z);obs.stream.close()
 assert not torch.equal(outputs[0],outputs[1])
 for name in ('source','predict','score','feature_check','audit','preflight','dispatch','publish'):importlib.import_module('experiments.cvs_receiver_vrex50.'+name)
 from .evidence import loss_close
 assert loss_close(180.93373107910156,180.93372641227415) and not loss_close(180.93373107910156,180.94373107910156) and not loss_close(float('nan'),1.)
 from .publish import configure,transport
 configure();compile(transport.REMOTE,'<actualVREXremote>','exec');compile(transport.INSPECT,'<actualVREXinspect>','exec')
 assert 'experiments.cvs_receiver_vrex50.preflight' in transport.REMOTE and 'experiments.cvs_receiver_vrex50.dispatch' in transport.REMOTE and "done['models']==3" in transport.INSPECT and "done['prediction_count']==504000" in transport.INSPECT
 return dict(status='PASS',focused_negatives=len(neg),exact_native_hook=True,formula_and_zero_penalty_and_missingRX=True,two_actual_initial_states_identical=True,original_U_loss_unchanged=True,native50_33_17_budget=True,actual_joint_optimizer_updates=body_updates,existing_W_joint_update=True,W_zeroing_does_not_change_joint160=True,actual_updated_feature_difference=True,real_IQ_or_checkpoint_read=False,head_fit_updates=0,synthetic_optimizer_steps_per_arm=1)
if __name__=='__main__':print(json.dumps(checks()))
