"""Actual wiring/fold/rank and new source-reference negatives; no data access."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json,tempfile
import torch
from . import design as d,model
from experiments.cvs_feature_restitution.features import layout
from experiments.cvs_time_factorial50 import model as tm

def check():
 torch.set_num_threads(2);c=d.config(d.rows()[1]);base=model.build(d.config(d.rows()[0]),'cpu');rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None;m=model.build(c,'cpu');core,prefix,_=layout(m)
 assert torch.equal(torch.get_rng_state(),rng)
 if cuda_rng is not None:assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
 for k,v in m.state_dict().items():
  if not any(k.startswith(prefix+'.'+layer+'.') for layer in ('t1','t2','t3')):assert torch.equal(v,base.state_dict()[k]),k
 counts=[sum(p.numel() for p in getattr(core,l).parameters()) for l in ('t1','t2','t3')];assert counts==[14640,28800,38304]
 compact=tm.build(d.compact_config(d.SEEDS[0]),'cpu');compact_core,_,_=layout(compact);assert sum(counts)-sum(p.numel() for l in ('t1','t2','t3') for p in getattr(compact_core,l).parameters())==72
 fold=[]
 for l in ('t1','t2','t3'):
  b=deepcopy(getattr(core,l)).double().eval();x=torch.randn(3,b.dw.in_channels,32,dtype=torch.float64,requires_grad=True);k=model.equivalent_kernel(b);a=b.preGN(x);f=torch.nn.functional.conv1d(x,k,padding=2);assert torch.allclose(a,f,atol=1e-12,rtol=1e-12)
  w=torch.randn_like(a);ga=torch.autograd.grad((a*w).sum(),x,retain_graph=True)[0];gf=torch.autograd.grad((f*w).sum(),x)[0];assert torch.allclose(ga,gf,atol=1e-12,rtol=1e-12);sv=torch.linalg.svdvals(k.detach().permute(1,0,2));assert int((sv>sv[:,0:1]*1e-10).sum(1).max())<=2;fold.append(dict(layer=l,preGN_error=float((a-f).detach().abs().max()),input_gradient_error=float((ga-gf).abs().max())))
 x=torch.randn(8,2,256,generator=torch.Generator().manual_seed(710));y=torch.arange(8)%6;m.eval()
 with torch.no_grad():
  z=m.forward_identity_only(x)['id_feat_joint'];h=model.classifier(m);w=h.weight.detach().clone();assert torch.equal(m.id_backbone.encoder.features(x),z);h.weight.zero_();assert torch.equal(m.id_backbone.encoder.features(x),z);del h.weight;assert torch.equal(m.id_backbone.encoder.features(x),z);h.weight=torch.nn.Parameter(w)
  clone=model.build(c,'cpu').eval();clone.load_state_dict(m.state_dict(),strict=True);assert torch.equal(clone(x),m(x));teacher=deepcopy(m).eval();assert torch.equal(teacher(x),m(x))
 m.train();before={n:p.detach().clone() for n,p in m.named_parameters()};args=d.make_args(c,'cpu')
 with m._phase1_labeled_feature_context():v=m(x,y_tx=y,return_aux=True)
 torch.nn.functional.cross_entropy(v['tx_logits'],y).backward();opt=torch.optim.AdamW(m.parameters(),lr=args.lr,weight_decay=args.weight_decay)
 for l in ('t1','t2','t3'):
  for n,p in getattr(core,l).named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,(l,n)
 opt.step();updates={l:sum(int((p!=before[n]).sum()) for n,p in m.named_parameters() if n.startswith(prefix+'.'+l+'.')) for l in ('t1','t2','t3')};assert all(n>0 for n in updates.values());assert not torch.equal(h.weight,before[prefix+'.cls_head.weight']);m.eval();assert not torch.equal(m.id_backbone.encoder.features(x),z)
 bad=deepcopy(c);bad['separate_head_training']=True
 try:d.validate(bad)
 except ValueError:pass
 else:raise AssertionError('Head-fitting config accepted')
 from experiments.cvs_feature_sequence50 import dispatch
 with tempfile.TemporaryDirectory() as td,patch.object(d,'BASE',Path(td)),patch.object(d,'source_reference_metrics',lambda s:dict(source_val_accuracy=.99,source_val_worst_rx=.97)):
  paths={}
  for r in d.rows()[:2]:
   cfg=d.config(r);p=Path(td)/r['row_id']/'source';p.mkdir(parents=True);paths[r['arm']]=p;d.write(p/'completion.json',dict(final_training_L_CE=.2,final_source_metrics=dict(source_val_accuracy=.98,source_val_worst_rx=.96)))
  b=d.baseline_config(d.SEEDS[0]);b['output_root']=str(paths['BASE_REF'])
  with patch.object(d,'baseline_config',lambda s:b):
   assert dispatch.eligible(d.rows()[2],design=d,proof=lambda c:paths[c['arm']])
   v=d.read(paths['SINGLE_BASIS']/'completion.json');v['final_source_metrics']['source_val_accuracy']=.979;d.write(paths['SINGLE_BASIS']/'completion.json',v)
   try:dispatch.eligible(d.rows()[2],design=d,proof=lambda c:paths[c['arm']])
   except ValueError:pass
   else:raise AssertionError('Compact source reference bypassed')
 return dict(status='PASS',block_parameters=counts,extra_parameters_to_compact=72,actual_operator_fold=fold,preGN_kernel_rank_bound=2,not_feature_rank=True,default_W_modified_deleted_invariant=True,strict_checkpoint_teacher_roundtrip=True,optimizer_gradient_numerical_updates=updates,actual_joint160_changed=True,synthetic_not_recognition_evidence=True,RNG_preserved=True,source_compact_reference_consumed=True,head_fit_config_rejected=True,target_read=False)
if __name__=='__main__':print(json.dumps(check()))
