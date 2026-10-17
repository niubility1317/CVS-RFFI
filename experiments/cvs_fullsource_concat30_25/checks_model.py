"""Exercise both full256 native forwards, joint gradients and clean-only SupCon."""
import copy,json
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .f_base.optimizer import build,clip_physical
def run():
 torch.set_num_threads(2);rng=torch.Generator().manual_seed(2203)
 x=torch.randn(128,2,256,generator=rng);y=torch.arange(128)%6
 rx=torch.tensor([d.RX[(i//3)%5] for i in range(128)]);days=torch.arange(128)%3+1
 dm=torch.tensor([d.RX.index(int(r))*3+int(day)-1 for r,day in zip(rx,days)]);domains=torch.cat((dm,dm));meta=dict(rx_i=rx)
 initial=None;reports=[]
 for row in d.rows():
  torch.manual_seed(d.SEED);m=api.build(d.config(row),'cpu').train();state={k:v.clone() for k,v in m.state_dict().items()}
  if initial is None:initial=state
  else:assert all(torch.equal(initial[k],v) for k,v in state.items())
  assert len(state)==206 and sum(p.numel() for p in m.parameters())==900722
  teacher=copy.deepcopy(m);assert teacher._full_base_scope.__self__ is teacher and teacher._source_labeled_feature_hook.__self__ is teacher
  z=torch.cat((x,x if row['arm']=='B00' else x*.99));seen=[];old=m.id_backbone.encoder.features
  def trace(v):seen.append(v.detach().clone());return old(v)
  m.id_backbone.encoder.features=trace;m._feature_l_steps=m._full_calls=1110
  with m._phase1_labeled_feature_context():out=m(z,return_aux=True)
  ce=F.cross_entropy(out['tx_logits'],torch.cat((y,y)),label_smoothing=.01)
  correction,stats=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(dm,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128)
  assert len(seen)==2 and torch.equal(seen[0],z) and torch.equal(seen[1],z) and m._full_pending is None
  assert abs(m._full_records[-1]['weight']-.01)<1e-12 and len(m._full_records[-1]['details'])==4
  assert all(q['shape'][0]==128 and q['total_anchors']==128 for q in m._full_records[-1]['details'])
  m.id_backbone.encoder.features=old
  head='id_backbone.encoder.core.id_backbone.cls_head.weight';before=m.get_parameter(head).clone()
  opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);(ce+correction).backward()
  assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())
  keys=[head]+[f'id_backbone.encoder.sequence_pool.views.{i}.local_projection.weight' for i in range(3)]+['id_backbone.encoder.core.id_backbone.sinc.low_coordinate']
  assert all(m.get_parameter(k).grad is not None and bool((m.get_parameter(k).grad!=0).any()) for k in keys)
  clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(before,m.get_parameter(head))
  m.eval()
  with torch.no_grad():assert m.forward_identity_only(x[:2])['tx_logits'].shape==(2,6)
  reports.append(dict(arm=row['arm'],actual_concat_shape=[256,2,256],same_IQ_RDrop=True,clean_only_SupCon=True,default_head_joint_update=True,finite_joint_gradients=True))
 return dict(status='PASS',synthetic_NN=1028,optimizer_updates=2,real_IQ=0,CP=0,all206_initial_tensors_equal=True,rows=reports)
if __name__=='__main__':print(json.dumps(run()))
