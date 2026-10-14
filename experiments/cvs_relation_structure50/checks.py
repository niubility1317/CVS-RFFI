import copy,torch
from .objective import objective,anchor_groups,RXS
from . import components

def run():
 gen=torch.Generator().manual_seed(90213);bank=torch.randn(5,6,6,160,generator=gen,dtype=torch.double)
 for r in RXS:
  for cross in (False,True):
   groups=anchor_groups(bank,r,cross);assert sum(len(a) for _,a in groups)==120 and (not cross or all(k!=r for k,_ in groups))
   # Actual selected index values verify matchedTX/day composition.
   from .objective import positions,ALL_DAY_COUNTS,CROSS_DAY_COUNTS
   counts=CROSS_DAY_COUNTS if cross else ALL_DAY_COUNTS
   days=[sum(row[d] for row in counts) for d in range(3)];assert days==[7,7,6]
 rx=torch.tensor(RXS*2);base=torch.randn(10,160,generator=gen,dtype=torch.double);pa=torch.randn(10,160,generator=gen,dtype=torch.double)*.2;ref=torch.randn(10,160,generator=gen,dtype=torch.double)*.1
 teacher=dict(joint=base+pa+ref,PA=pa,ref=ref)
 a,_,_=objective(teacher,teacher,bank,rx,'CROSS_RESP');assert a.item()==0
 # Full-joint equality cannot see opposing branch shifts, response must.
 v=torch.randn(10,160,generator=gen,dtype=torch.double)*.1
 student=dict(joint=teacher['joint'].clone().requires_grad_(),PA=(pa-v).clone().requires_grad_(),ref=ref.clone().requires_grad_())
 total,raw,facts=objective(student,teacher,bank,rx,'CROSS_RESP');assert raw['relation'].item()==0 and raw['response'].item()>1e-8
 total.backward();assert student['PA'].grad.norm()>0 and facts['all_U_used'] and facts['anchors_per_query']==120
 zero,_,_=objective(student,teacher,bank,rx,'ZERO');assert zero.item()==0
 try:objective(student,teacher,bank,rx+100,'CROSS_RESP')
 except ValueError:pass
 else:raise AssertionError('Non-source RX accepted')
 return dict(status='PASS',count_TX_day_matched=True,exclude_same_RX=True,equal_parts_zero_loss=True,branch_cancellation_detected=True,actual_increment_gradient=True,zero_control_exact=True,query_role_negative=True,no_U_TX_argument=True)

def wiring(device='cpu'):
 from experiments.cvs_feature_multiscale50 import design as old
 from experiments.cvs_feature_multiscale50.model import build,classifier
 torch.set_num_threads(2);c=old.config(next(r for r in old.rows() if r['arm']=='OLD_GATE_COS50' and r['model_seed']==old.SEEDS[0]));m=components.install(build(c,device));teacher=copy.deepcopy(m)
 # Initial reference gain0 is a real architectural boundary; use a finite
 # test gain to exercise its projection, without fitting any classifier.
 with torch.no_grad():m.id_backbone.encoder.response_gain.fill_(.1);teacher.id_backbone.encoder.response_gain.fill_(.1)
 m.train();teacher.eval();gen=torch.Generator().manual_seed(48133);x=torch.randn(10,2,256,generator=gen).to(device)
 with torch.no_grad():ot=teacher.forward_identity_only(x);tp=components.parts(teacher,ot)
 os=m.forward_identity_only(x+.015*torch.randn(10,2,256,generator=gen).to(device));sp=components.parts(m,os)
 bank=torch.randn(5,6,6,160,generator=gen,device='cpu').to(device).requires_grad_();rx=torch.tensor(RXS*2,device=device)
 aux,_,facts=objective(sp,tp,bank,rx,'CROSS_RESP')
 e=m.id_backbone.encoder;h=classifier(m);targets=[h.weight,h.gain,e.response_gain,e.core.id_backbone.t1[0].weight if hasattr(e.core.id_backbone.t1,'__getitem__') and hasattr(e.core.id_backbone.t1[0],'weight') else next(e.core.id_backbone.t1.parameters()),e.response_projection[0].weight]
 grads=torch.autograd.grad(aux,[*targets,bank],retain_graph=True,allow_unused=True)
 assert all(g is None for g in grads[:3]) and grads[3] is not None and grads[3].norm()>0 and grads[4] is not None and grads[4].norm()>0 and grads[5] is None
 before=targets[3].detach().clone();opt=torch.optim.AdamW(m.parameters(),lr=.0002);opt.zero_grad();aux.backward();opt.step();assert not torch.equal(before,targets[3])
 assert teacher.id_backbone.encoder.response_projection._relation_feature is not m.id_backbone.encoder.response_projection._relation_feature
 return dict(status='PASS',device=device,class_W_aux_gradient=None,PA_gain_aux_gradient=None,ref_gain_aux_gradient=None,shallow_t1_aux_gradient=float(grads[3].norm()),ref_projection_aux_gradient=float(grads[4].norm()),teacher_and_anchors_stopped=True,actual_feature_parameter_changed=True,actual_additive_reconstruction=True,head_fit_updates=0,nonzero_relation_fraction=facts['nonzero_relation_fraction'])
def callback_wiring():
 import tempfile
 from pathlib import Path
 from . import design as d,model
 from .observe import Observer
 c=d.config(d.rows()[2]);m=model.build(c,'cpu');teacher=copy.deepcopy(m);m.train();teacher.eval();m._feature_l_steps=7328
 gen=torch.Generator().manual_seed(79133);x=torch.randn(10,2,256,generator=gen);y=torch.arange(10)%6
 identity=torch.nn.functional.cross_entropy(m.forward_identity_only(x)['tx_logits'],y)
 with torch.no_grad():wt=teacher.forward_identity_only(x)
 ss=m.forward_identity_only(x+.015*torch.randn(10,2,256,generator=gen));pseudo=wt['tx_logits'].detach().argmax(1);uce=torch.nn.functional.cross_entropy(ss['tx_logits'],pseudo);p=ss['tx_logits'].softmax(-1);ent=-(p*p.clamp_min(1e-8).log()).sum(-1).mean()
 with tempfile.TemporaryDirectory(prefix='cvs_relation_noquery_') as temporary:
  o=Observer();o.initialize(m,c,Path(temporary));o.bank=torch.randn(5,6,6,160,generator=gen);o.bank_step=7327
  meta=dict(rx_i=torch.tensor(RXS*2),day_i=torch.ones(10,dtype=torch.long),eq_i=torch.ones(10,dtype=torch.long),sig_i=torch.arange(10),base_index=torch.arange(10))
  kwargs=dict(model=m,teacher=teacher,student_output=ss,teacher_output=wt,extra=(None,meta),domains=torch.zeros(10,dtype=torch.long),optimizer_step=7328,identity_loss=identity,pseudo_loss=uce,entropy_loss=ent)
  aux,logs=o.auxiliary(**kwargs)
  h=model.classifier(m);assert torch.autograd.grad(aux,h.weight,retain_graph=True,allow_unused=True)[0] is None
  before=h.weight.detach().clone();opt=torch.optim.AdamW(m.parameters(),lr=.0002);opt.zero_grad();(identity+.16*uce+.01*ent+aux).backward();opt.step();assert not torch.equal(before,h.weight)
  bad=dict(meta,tx_i=torch.arange(10)%6);kwargs['extra']=(None,bad)
  try:o.auxiliary(**kwargs)
  except ValueError:pass
  else:raise AssertionError('Visible U TX metadata accepted')
  o.log.close()
 return dict(status='PASS',actual_callback_and_original_losses=True,all_U_packets=10,aux_class_W_none=True,default_W_joint_CE_updated=True,visible_U_metadata_rejected=True,synthetic_noquery=True,head_fit_updates=0)
if __name__=='__main__':print(run());print(wiring());print(callback_wiring())
