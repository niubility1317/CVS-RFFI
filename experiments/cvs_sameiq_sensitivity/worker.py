import json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .loss import row,direction,geometry
from experiments.cvs_F_nontrue20 import design as baseline
from experiments.cvs_F_nontrue20.model import own_model
from experiments.cvs_F_nontrue20.teacher import load as load_teacher
from experiments.cvs_F_nontrue20.quality import panel
from experiments.cvs_F_nontrue20.channel import tensor_iq
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity
def select(data):
 groups={}
 for i,key in enumerate(zip(data['y'].tolist(),data['rx'].tolist(),data['day'].tolist())):groups.setdefault(tuple(key),[]).append(i)
 if len(groups)!=90 or any(len(v)!=20 for v in groups.values()):raise ValueError('Original balanced source panel differs')
 selected=[i for k,v in sorted(groups.items()) for i in v[:4]]
 # Interleave the four fixed within-cell entries: each90 batch covers all90 cells.
 return torch.tensor([selected[cell*4+j] for j in range(4) for cell in range(90)],dtype=torch.long)
def qualify(records):
 urban=[r for r in records if r['view']!='clean'];clean=[r for r in records if r['view']=='clean']
 def avg(rs,key):
  xs=[r['body'][key]['cosine'] for r in rs if r['body'][key]['cosine'] is not None]
  return sum(xs)/len(xs) if len(xs)==len(rs) and xs else None
 t=avg(urban,'true_CE');w=avg(urban,'wrong_CE');cl=avg(clean,'true_CE');adv=100*sum(r['teacher_correct']-r['student_correct'] for r in urban)/(d.N*4)
 passed=t is not None and w is not None and cl is not None and t>=0. and t-w>=.01 and cl>=-.05 and adv>=1.
 return dict(rationale_pass=passed,urban_teacher_advantage_pp=adv,urban_body_true_CE_cos=t,urban_body_wrong_CE_cos=w,urban_body_true_minus_wrong=None if t is None or w is None else t-w,clean_body_true_CE_cos=cl,automatic_training=False,E50_qualified=False,recognition_benefit_proved=False,decision='SOURCE_RATIONALE_SUPPORTS_FIXED_E20_PILOT_ONLY' if passed else 'CLOSE_FIXED_SIGNED_TRUE_ROW_RATIONALE; no threshold/label/norm rescue')
def run():
 d.validate(d.config());start=time.perf_counter();out=d.BASE;torch.set_num_threads(2);nn={'student':0,'teacher':0}
 if os.geteuid()==0:raise ValueError('Ordinary account required')
 def zs(m,x):nn['student']+=len(x);return m.forward_identity_only(x)['tx_logits']
 def zt(m,x):nn['teacher']+=len(x);return m(x)['tx_logits']
 try:
  with lock(out/'owner.lock',nonblocking=True):
   gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
   with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
    if len(capacity()[gpu]['pids']|{os.getpid()})>2:raise ValueError('Global GPU process cap2 includes other owners')
    torch.cuda.init()
   with numerical_context(baseline.FULL_FP32_POLICY):
    c=baseline.config(next(r for r in baseline.rows() if r['arm']=='B1'));m=own_model(c,'cuda:0').requires_grad_(True);teacher=load_teacher('cuda:0');snap={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};tsnap={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()};named=list(m.named_parameters());params=[p for n,p in named]
    # One real-checkpoint no-query smoke, including the actual second-order path.
    with torch.random.fork_rng(devices=[0]):
     torch.manual_seed(2026101931);x=torch.randn(2,2,256,device='cuda:0',requires_grad=True);t=zt(teacher,x);y=t.detach().argmax(1);gt=row(t,x,y).detach();m.train();xs=x.detach().clone().requires_grad_(True);s=zs(m,xs);gs=row(s,xs,y,create_graph=True,retain_graph=True);pen,_=direction(gs,gt,torch.ones(2,dtype=torch.bool,device=x.device));grads=torch.autograd.grad(pen,params,allow_unused=True)
     if not torch.isfinite(s).all() or any(g is not None and not torch.isfinite(g).all() for g in grads) or not any(g is not None and bool(g.abs().sum()>0) for g in grads):raise ValueError('Actual checkpoint double-backward smoke failed')
    m.eval();d.write(out/'checkpoint_smoke.json',dict(status='PASS',NN=sum(nn.values()),query_access=False,source_IQ_opened=False,actual_student_epoch=20,actual_teacher_epoch=200,default_head_only=True,optimizer_updates=0));print('SMOKE PASS actual F/CSIL and second-order body/default-head; no query',flush=True)
    data,roles=panel(torch);ix=select(data);yall=data['y'][ix].long();ids=[data['ids'][int(i)] for i in ix];raw=data['x'][ix];observations=np.load(baseline.PANEL/'observations/iq.npy',mmap_mode='r',allow_pickle=False)
    if len(ids)!=360 or len(set(ids))!=360 or not set(ids)<=set(baseline.read(baseline.SOURCE)['role_ids']['V']):raise ValueError('Fixed source V360 membership differs')
    d.write(out/'data_roles.json',dict(role='V',ids=ids,source_RX=[1,3,4,6,8],days=[1,2,3],per_TX_RX_day=4,U_truth_read=False,target_access=False,fit_updates=0))
    records=[];saved={};nix=ix.tolist()
    for view in d.VIEWS:
     if view=='clean':inputs=raw
     else:
      draw=int(view[1]);scene=1 if view.endswith('mid') else 2;inputs=tensor_iq(observations[draw,scene,0,nix],'cpu')
     values={k:[] for k in ('teacher','student','teacher_plus','teacher_minus','student_plus','student_minus','g_teacher','g_wrong','g_student')}
     for begin in range(0,360,90):
      y=yall[begin:begin+90].to('cuda:0');x=inputs[begin:begin+90].to('cuda:0');xt=x.detach().clone().requires_grad_(True);t=zt(teacher,xt);mask=t.detach().argmax(1)==y;gt=row(t,xt,y,retain_graph=True).detach();gw=row(t,xt,(y+1)%6).detach();xs=x.detach().clone().requires_grad_(True);s=zs(m,xs);gs=row(s,xs,y,create_graph=True,retain_graph=True)
      pen,cos=direction(gs,gt,mask);wrong,_=direction(gs,gw,mask);ce=torch.nn.functional.cross_entropy(s,y,label_smoothing=.01)
      gJ=torch.autograd.grad(pen,params,allow_unused=True,retain_graph=True);gW=torch.autograd.grad(wrong,params,allow_unused=True,retain_graph=True);gCE=torch.autograd.grad(ce,params,allow_unused=True)
      indices={key:[i for i,(n,p) in enumerate(named) if ('.cls_head.' in n)==(key=='head')] for key in ('body','head')}
      grouped={key:dict(true_CE=geometry([named[i] for i in inds],[gJ[i] for i in inds],[gCE[i] for i in inds]),wrong_CE=geometry([named[i] for i in inds],[gW[i] for i in inds],[gCE[i] for i in inds])) for key,inds in indices.items()}
      if not torch.isfinite(gt).all() or not torch.isfinite(gw).all() or not torch.isfinite(gs).all():raise FloatingPointError('Nonfinite source input derivative')
      u=torch.nn.functional.normalize(gt.flatten(1),dim=1,eps=1e-12).reshape_as(x);h=.001*x.flatten(1).norm(dim=1).clamp_min(1e-6);delta=h[:,None,None]*u
      with torch.no_grad():tp=zt(teacher,x+delta);tm=zt(teacher,x-delta);sp=zs(m,x+delta);sm=zs(m,x-delta)
      exact=h*gt.flatten(1).norm(dim=1);observed=(tp.gather(1,y[:,None])-tm.gather(1,y[:,None])).squeeze(1)/2;rel=(observed-exact).abs()/(exact.abs()+1e-8)
      rec=dict(view=view,begin=begin,n=90,teacher_correct=int(mask.sum()),student_correct=int((s.detach().argmax(1)==y).sum()),teacher_correct_student_wrong=int((mask&(s.detach().argmax(1)!=y)).sum()),teacher_wrong_student_correct=int((~mask&(s.detach().argmax(1)==y)).sum()),penalty=float(pen.detach()),wrong_penalty=float(wrong.detach()),student_CE=float(ce.detach()),input_true_wrong_cosine=float(torch.nn.functional.cosine_similarity(gt.flatten(1),gw.flatten(1),dim=1,eps=1e-12).mean()),student_teacher_input_cosine=float(cos.detach().mean()),teacher_finite_difference_relative_error_mean=float(rel.mean()),**grouped)
      rec.update(teacher_zero_input_derivatives=int((gt.flatten(1).norm(dim=1)==0).sum()),student_zero_input_derivatives=int((gs.detach().flatten(1).norm(dim=1)==0).sum()))
      records.append(rec)
      for key,value in dict(teacher=t,student=s,teacher_plus=tp,teacher_minus=tm,student_plus=sp,student_minus=sm,g_teacher=gt,g_wrong=gw,g_student=gs).items():values[key].extend(value.detach().cpu().tolist())
      print(json.dumps(dict(view=view,complete=begin+90,NN=sum(nn.values()),updates=0)),flush=True)
     saved.update({view+'_'+key:np.asarray(value,dtype=np.float32) for key,value in values.items()})
    if sum(nn.values())!=10804:raise ValueError('Fixed source derivative/finite-difference forward count differs')
    if any(p.grad is not None for p in params) or any(p.grad is not None or p.requires_grad for p in teacher.parameters()) or any(not torch.equal(v,m.state_dict()[k].detach().cpu()) for k,v in snap.items()) or any(not torch.equal(v,teacher.state_dict()[k].detach().cpu()) for k,v in tsnap.items()):raise ValueError('Frozen diagnostic changed weights/buffers or populated parameter grads')
    np.savez_compressed(out/'source_derivatives.npz',ids=np.asarray(ids),TX=yall.tolist(),RX=data['rx'][ix].tolist(),day=data['day'][ix].tolist(),**saved)
    d.write(out/'batch_geometry.json',dict(records=records,query_access=False,parameter_gradients_not_applied=True));decision=qualify(records);d.write(out/'source_decision.json',decision)
    d.write(out/'complete.json',dict(status='SOURCE_DIAGNOSED',config=d.config(),unique_source_V=360,NN=nn,total_NN=sum(nn.values()),counterfactual_source_packets=7200,new_physical_draws=0,optimizer_updates=0,head_fit_updates=0,U_truth_read=False,target_access=False,weights_and_buffers_unchanged=True,parameter_grads_none=True,student_batchnorm_modules=sum(isinstance(v,torch.nn.modules.batchnorm._BatchNorm) for v in m.modules()),wall_seconds=time.perf_counter()-start,decision=decision,goal_complete=False));print(json.dumps(decision),flush=True)
 except BaseException as error:
  d.write(out/'failure.json',dict(status='TECHNICAL_FAILED',error=str(error),traceback=traceback.format_exc(),outputs_preserved=True,no_automatic_retry=True));raise
