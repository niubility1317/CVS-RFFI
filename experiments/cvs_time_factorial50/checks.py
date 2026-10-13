"""Numerical wiring checks and protocol negatives, no real dataset/target access."""
from copy import deepcopy
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from . import design as d,model
from .diagnostics import equivalent_kernel,remove,measure_update
from experiments.cvs_feature_restitution.features import layout
from experiments.cvs_stage2_decouple.common import parameter_groups

def check_model():
 torch.set_num_threads(2);x=torch.randn(8,2,256,generator=torch.Generator().manual_seed(710));y=torch.arange(8)%6
 base=model.build(d.config(d.rows()[0]),'cpu').eval();head=deepcopy(model.classifier(base).state_dict());base_state=base.state_dict();base_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None;records=[];states={};counts=[]
 for row in d.rows()[1:5]:
  c=d.config(row);m=model.build(c,'cpu');assert torch.equal(torch.get_rng_state(),base_rng),'new initialization consumed global RNG'
  if cuda_rng is not None:assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())), 'new initialization changed CUDA RNG'
  h=model.classifier(m);assert all(torch.equal(v,head[k]) for k,v in h.state_dict().items());core,prefix,_=layout(m)
  for n,v in m.state_dict().items():
   if not any(n.startswith(prefix+'.'+l+'.') for l in ('t1','t2','t3')):assert torch.equal(v,base_state[n]),n
  states[row['arm']]={n:v.clone() for n,v in m.state_dict().items()};counts.append(sum(p.numel() for l in ('t1','t2','t3') for p in getattr(core,l).parameters()))
  m.eval();v=m.forward_identity_only(x);z=v['id_feat_joint'].detach();pure=m.id_backbone.encoder.features(x).detach();assert torch.equal(z,pure)
  W=h.weight.detach().clone()
  with torch.no_grad():h.weight.copy_(torch.randn_like(h.weight))
  assert torch.equal(m.id_backbone.encoder.features(x),z)
  del h.weight;assert torch.equal(m.id_backbone.encoder.features(x),z);h.weight=torch.nn.Parameter(W)
  clone=model.build(c,'cpu').eval();clone.load_state_dict(m.state_dict(),strict=True);assert torch.equal(clone.forward_identity_only(x)['id_feat_joint'],z)
  teacher=deepcopy(m).eval();assert torch.equal(teacher.forward_identity_only(x)['id_feat_joint'],z)
  m.train();before={n:p.detach().clone() for n,p in m.named_parameters()}
  with m._phase1_labeled_feature_context():out=m(x,y_tx=y,return_aux=True)
  torch.nn.functional.cross_entropy(out['tx_logits'],y).backward();pg,_=parameter_groups(m);optimizer=torch.optim.AdamW(pg,lr=.0002)
  for l in ('t1','t2','t3'):
   for n,p in getattr(core,l).named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,(row,l,n)
  assert h.weight.grad is not None and h.weight.grad.norm()>0;optimizer.step()
  changed={l:sum(int((p!=before[n]).sum()) for n,p in m.named_parameters() if n.startswith(prefix+'.'+l+'.')) for l in ('t1','t2','t3')};assert all(v>0 for v in changed.values())
  m.eval();after=m.forward_identity_only(x)['id_feat_joint'];assert not torch.equal(z,after)
  # L context is counted only for real labeled student; U/EMA calls do not increment it.
  assert m._feature_l_steps==1;m.train();m(x,y_tx=None,return_aux=True);teacher(x,y_tx=None,return_aux=True);assert m._feature_l_steps==1
  records.append(dict(arm=row['arm'],temporal_parameters=counts[-1],changed_values_by_layer=changed,default_W_joint_updated=not torch.equal(h.weight,before[prefix+'.cls_head.weight']),actual_joint160_changed=True,W_modified_deleted_feature_invariant=True,strict_checkpoint_teacher_roundtrip=True,RNG_preserved=True,CUDA_RNG_preserved=True if cuda_rng is not None else 'CUDA_UNAVAILABLE'))
 assert set(counts)=={81672}
 for arm,state in states.items():
  for n,v in state.items():assert torch.equal(v,states['COMPACT_LINEAR'][n]),('initial weights differ',arm,n)
 # Same initialized weights, differing outputs for dilation/nonlinearity are intentional.
 return dict(status='PASS',rows=records,extra_parameters=60672,default_head_only=True,head_fit_updates=0,target_read=False)

def check_math():
 # Use real block definitions from installed model, not another recreated operator.
 c=d.config(d.rows()[1]);m=model.build(c,'cpu');core,_,_=layout(m);errors=[]
 for l in ('t1','t2','t3'):
  b=getattr(core,l).double().eval();x=torch.randn(3,b.branches[0].in_channels,32,dtype=torch.float64,requires_grad=True);k=equivalent_kernel(b);raw=b.fuse(torch.cat([p(x) for p in b.branches],1))+b.skip(x);fold=torch.nn.functional.conv1d(x,k,padding=k.shape[-1]//2)
  assert torch.allclose(raw,fold,atol=1e-12,rtol=1e-12);weight=torch.randn_like(raw);g=torch.autograd.grad((raw*weight).sum(),x,retain_graph=True)[0];gf=torch.autograd.grad((fold*weight).sum(),x)[0];assert torch.allclose(g,gf,atol=1e-12,rtol=1e-12);errors.append(dict(layer=l,preGN_max=float((raw-fold).detach().abs().max()),input_gradient_max=float((g-gf).abs().max())))
 b=getattr(core,'t1');b.nonlinear=True
 try:equivalent_kernel(b)
 except ValueError:pass
 else:raise AssertionError('Nonlinear branch folded')
 return dict(status='PASS',actual_operator_fold=errors,nonlinear_fold_rejected=True)

def check_pipeline():
 from experiments.cvs_feature_sequence50.checks_pipeline import check as previous
 result=previous()
 from . import dispatch,provenance
 with tempfile.TemporaryDirectory() as td,patch.object(d,'BASE',Path(td)),patch.object(dispatch,'source_proof',lambda c:Path(td)/c['row_id']/'source'):
  later=d.rows()[5];assert not dispatch.eligible(later)
  for r in d.rows()[:5]:
   p=Path(td)/r['row_id']/'source';p.mkdir(parents=True);d.write(p/'completion.json',dict(final_training_L_CE=.2,final_source_metrics=dict(source_val_accuracy=.98,source_val_worst_rx=.94)))
  with patch.object(Path,'is_file',lambda p:True):
   assert dispatch.eligible(later);p=Path(td)/d.rows()[1]['row_id']/'source/completion.json';v=d.read(p);v['final_source_metrics']['source_val_accuracy']=.96;d.write(p,v)
   try:dispatch.eligible(later)
   except ValueError:pass
   else:raise AssertionError('Source screen allowed failure')
 return dict(status='PASS',shared_default_pipeline=result,new_source_gate=True)

def check_observer():
 from .observe import Observer
 from experiments.cvs_phase1_stack.runtime import installed
 c=d.config(d.rows()[1]);m=model.build(c,'cpu');obs=Observer();obs.initialize(m,c,Path('.'));args=d.make_args(c,'cpu')
 before={k:v.clone() for k,v in m.state_dict().items()}
 with installed(c) as n:
  metadata=dict(rx_i=torch.tensor([1,1]),day_i=torch.tensor([0,0]),sig_i=torch.tensor([1,2]),tx_label_visible=torch.tensor([False,False]))
  _,sentinels,extra=n.move_batch((torch.zeros(2,2,256),torch.tensor([-1,-1]),torch.tensor([0,0]),metadata),torch.device('cpu'))
  assert isinstance(extra,tuple) and (sentinels==-1).all();pseudo=torch.tensor([0,0]);conf=torch.tensor([.98,.98]);domains=torch.tensor([0,0])
  expected_conf=n._threshold_mask(conf,domains,args);expected_time=n._temporal_mask_tensor(pseudo,conf,extra,args,torch.device('cpu'));original=n._temporal_mask_tensor
  with obs.native_scope(n):
   assert torch.equal(n._threshold_mask(conf,domains,args),expected_conf)
   assert torch.equal(n._temporal_mask_tensor(pseudo,conf,extra,args,torch.device('cpu')),expected_time)
   m.train();obs.collect(m,(),dict(y_tx=None),dict(tx_logits=torch.tensor([[2.,0.,0.,0.,0.,0.],[2.,0.,0.,0.,0.,0.]])))
  assert n._temporal_mask_tensor is original and obs.handle is None and obs.pending is None
  assert obs.groups['1/0']['accepted']==int((expected_conf&expected_time).sum()) and obs.groups['1/0']['total']==2
  measure_update(m,before)
  contaminated=(extra[0],dict(extra[1],tx_i=torch.tensor([0,0])))
  with obs.native_scope(n):
   n._threshold_mask(conf,domains,args)
   try:n._temporal_mask_tensor(pseudo,conf,contaminated,args,torch.device('cpu'))
   except ValueError:pass
   else:raise AssertionError('U TX metadata accepted')
 return dict(status='PASS',real_move_batch_tuple_interface=True,real_threshold_temporal_results_unchanged=True,actual_student_hook=True,U_TX_metadata_rejected=True,model_params_buffers_unchanged=True)
if __name__=='__main__':print(json.dumps(dict(math=check_math(),model=check_model(),pipeline=check_pipeline(),observer=check_observer())))
