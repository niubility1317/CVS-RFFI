import copy,json,sys,torch
from torch.nn import functional as F
from . import design as d,model as api,mechanism as f
def reject(fn):
 try:fn()
 except (ValueError,RuntimeError,TypeError,KeyError):return
 raise AssertionError('Illegal source RDrop action accepted')
def main():
 torch.set_num_threads(2);torch.manual_seed(19);negative=0
 y=torch.arange(48)//8;rx=torch.tensor(d.FIT).repeat_interleave(2).repeat(6)
 a=(torch.randn(48,6,dtype=torch.float64)*3).requires_grad_();b=torch.randn_like(a,requires_grad=True)
 original=api.original.terms(a,b,y,1.)[2]
 parts={};transports={}
 for arm in d.ARMS:
  P,meta=f.transport(y,rx,arm,a.dtype);transports[arm]=P
  assert meta['selected']==48 and meta['mass_error']<1e-14
  assert torch.equal(P,P.T) and torch.equal((P>0).sum(1),torch.ones(48,dtype=torch.long))
  assert meta['transported']==(48 if arm in ('SAME_RX','CROSS_RX') else 0)
  binary,relation=f.components(a,b,y,P,True);parts[arm]=(binary,relation)
  ga,gb=torch.autograd.grad(relation,(a,b),retain_graph=True)
  assert torch.equal(ga.gather(1,y[:,None]),torch.zeros(48,1,dtype=a.dtype)) and torch.equal(gb.gather(1,y[:,None]),torch.zeros(48,1,dtype=a.dtype))
  assert torch.allclose(ga.sum(1),torch.zeros(48,dtype=a.dtype),atol=1e-15) and torch.allclose(gb.sum(1),torch.zeros(48,dtype=a.dtype),atol=1e-15)
  audit=f.new_audit();loss,metrics=f.consistency(a,b,y,rx,arm,audit)
  assert torch.isfinite(loss) and audit['calls']==1 and audit['max_chain_error']<1e-14
  if arm=='FULL':
   assert torch.equal(loss,original)
   assert all(torch.equal(x,z) for x,z in zip(torch.autograd.grad(loss,(a,b),retain_graph=True),torch.autograd.grad(original,(a,b),retain_graph=True)))
  perm=torch.tensor([2,5,0,3,1,4]);inverse=torch.argsort(perm)
  pl,_=f.consistency(a[:,perm],b[:,perm],inverse[y],rx,arm,f.new_audit())
  assert torch.allclose(loss,pl,atol=1e-14,rtol=1e-14)
 assert all(torch.equal(z[0],parts['FULL'][0]) for z in parts.values())
 remap=dict(zip(d.FIT,reversed(d.FIT)));remapped=torch.tensor([remap[int(v)] for v in rx])
 assert torch.equal(f.transport(y,remapped,'CROSS_RX',a.dtype)[0],transports['CROSS_RX'])
 fallback=f.transport(torch.arange(6),torch.full((6,),3),'CROSS_RX',a.dtype)
 assert torch.equal(fallback[0],torch.eye(6,dtype=a.dtype)) and fallback[1]['selected']==0
 for labels,receivers,arm in [(y-1,rx,'FULL'),(y+1,rx,'FULL'),(y.float(),rx,'FULL'),(y,rx.float(),'FULL'),(y,rx[:1],'FULL'),(y,rx*0+1,'FULL'),(y,rx*0+9,'FULL'),(y,rx,'INVALID')]:reject(lambda:f.transport(labels,receivers,arm,a.dtype));negative+=1
 reject(lambda:f.consistency(a*float('nan'),b,y,rx,'FULL',f.new_audit()));negative+=1
 from .provenance import validate_args
 for row in d.rows():
  c=d.config(row);args=vars(d.make_args(c,'cpu'));validate_args(args,c)
  for key,value in [('use_aug',False),('label_smoothing',0.),('source_fold_held_rx',None),('source_fold_training_receivers','1,3,4,6,8'),('baseline_ckpt','unknown.pt'),('teacher_ckpt','unknown.pt'),('epochs',50)]:
   bad=dict(args);bad[key]=value;reject(lambda:validate_args(bad,c));negative+=1
  for key,value in [('model_seed',2026092702),('test_views',['clean']),('head_fit_updates',1),('fold',{})]:
   bad=dict(c);bad[key]=value;reject(lambda:d.validate(bad));negative+=1
 device='cuda:0' if '--cuda' in sys.argv else 'cpu'
 if device!='cpu':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 models=[api.build(d.config(row),device) for row in d.rows()];states=[m.state_dict() for m in models]
 assert all(set(s)==set(states[0]) and all(torch.equal(v,s[k]) for k,v in states[0].items()) for s in states[1:])
 torch.manual_seed(2026101581);x=torch.randn(128,2,256,device=device);aug=x*.7+.13;y=(torch.arange(128,device=device)//8)%6
 domains=(torch.arange(128,device=device)%8//2)*3+torch.arange(128,device=device)%3
 mapping={i*3+j:r for i,r in enumerate(d.FIT) for j in range(3)}
 get_rng=torch.get_rng_state if device=='cpu' else torch.cuda.get_rng_state;ends=[];CE=[];logits=[];stats={};norms={}
 for row,m in zip(d.rows(),models):
  m._evidence_domain_rx=mapping.copy();m.train();torch.manual_seed(2026101582)
  with m._phase1_labeled_feature_context():out=m(aug,y_tx=y,domain_labels=domains,return_aux=True)
  assert torch.equal(m._legacy_actual_L_input,aug)
  ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
  metadata=(domains.clone(),dict(rx_i=torch.tensor([mapping[int(v)] for v in domains.cpu().tolist()],device=device)))
  correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=metadata,domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128)
  assert m._legacy_actual_L_input is None and m._evidence_calls==m._legacy_calls==m._feature_l_steps==1 and m._legacy_extra_packets==128
  named=dict(m.named_parameters());names=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight','id_backbone.encoder.sequence_pool.local_projection.weight')
  gs=torch.autograd.grad(ce+correction,[named[n] for n in names]+[api.classifier(m).weight])
  assert all(torch.isfinite(g).all() and g.norm()>0 for g in gs)
  ends.append(get_rng().clone());CE.append(metrics['rdrop_evidence/paired_CE']);logits.append(out['tx_logits'].detach());stats[row['arm']]=m._evidence_audit.copy();norms[row['arm']]=[float(g.norm()) for g in gs]
  teacher=copy.deepcopy(m).eval();assert teacher._legacy_original_labeled_scope.__self__ is teacher and teacher._legacy_actual_L_input is None
  reject(lambda:teacher._source_labeled_feature_hook(model=teacher,labeled_output=out,labels=y,extra={},domains=domains,optimizer_step=2,identity_loss=ce,labeled_count=128));negative+=1
 assert all(torch.equal(v,CE[0]) for v in CE[1:]) and all(torch.equal(v,logits[0]) for v in logits[1:]) and all(torch.equal(v,ends[0]) for v in ends[1:])
 assert stats['SAME_RX']['selected_anchors']==stats['CROSS_RX']['selected_anchors']==stats['SAME_RX']['transported_anchors']==stats['CROSS_RX']['transported_anchors']
 fixture={arm:dict(source_health=True,selected_anchor_fraction=.3,exports={'FINAL':dict(count=1260,errors=100,mean_CE=1.)}) for arm in d.ARMS}
 assert not d.qualify(fixture)['qualified'];fixture['CROSS_RX']['exports']['FINAL']['errors']=99;assert d.qualify(fixture)['qualified']
 for key in ('CE','signal','health','same_RX'):
  bad=copy.deepcopy(fixture)
  if key=='CE':bad['CROSS_RX']['exports']['FINAL']['mean_CE']=1.01
  elif key=='signal':bad['CROSS_RX']['selected_anchor_fraction']=.09
  elif key=='health':bad['FULL']['source_health']=False
  else:bad['SAME_RX']['exports']['FINAL']['errors']=98
  assert not d.qualify(bad)['qualified']
 print(json.dumps(dict(status='PASS',device=device,negative_cases=negative,qualification_negative_cases=5,FP64_KL_chain_and_original_gradient=True,binary_term_unchanged=True,relation_true_class_logit_gradient_exactly_zero=True,class_and_RX_permutation_invariance=True,same_RX_cross_RX_transport_mass_matched=True,actual_initial_state_tensors=len(states[0]),actual_CE_and_logits_bitwise_equal=True,native_RNG_equal=True,post_augmentation_input_reused=True,actual_parameter_gradient_norms=norms,evidence_audits=stats,synthetic_identity_forwards=1024,real_IQ_reads=0,checkpoint_loads=0,optimizer_steps=0,head_fit_updates=0,target_access=False)))
if __name__=='__main__':main()
