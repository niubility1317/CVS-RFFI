"""Fixed E20+10 L-only three-term feature adaptation; existing W jointly updates with features."""
import gzip,json,math,time
import numpy as np
import torch
from . import design as d,model as api,optimizer as optim
from experiments.cvs_feature_adapters30.bank import open_bank
from experiments.cvs_feature_adapters30.loss import objective
from experiments.cvs_feature_adapters30.diagnose import readout
from experiments.cvs_F_nontrue20.channel import tensor_iq
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def train(c):
 d.validate(c);torch.set_num_threads(2);out=d.BASE/c['row_id']/'source';out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 with numerical_context(d.FULL_FP32_POLICY):
  parent=api.parent('cuda:0');parent_state={k:v.detach().clone() for k,v in parent.state_dict().items()};m=api.from_state(c,parent_state,'cuda:0');counts=api.counts(m);trainable=set(counts['trainable_names']);fixed={k:v for k,v in parent_state.items() if k not in trainable};opt=optim.build(m,c)
  L,indices,views=open_bank();initial=readout(m,out/'initial_source_readout');m.id_backbone.encoder._full_eval_packets=0;m.id_backbone.encoder.core.id_backbone.raw_temporal.packets=0
  d.write(out/'resolved_config.json',dict(config=c,**counts,backend_flags=actual_flags(),actual_mode='eval with student autograd',U_branch_executed=False,EMA_created=False,teacher='same frozen B0 clean parent only',new_classifier=False))
  d.write(out/'initialization.json',dict(parent_config=d.parent_config(),parent_run=d.REFERENCE_RUN,parent_row=d.PARENT_ROW,parent_checkpoint='source/final_ssdg.pth',parent_epochs=20,inheritance_verified=True,source_roles_match=True,head_frozen=False,head_joint=True,optimizer_reset=True,EMA_resumed=False,target_contact=False))
  d.write(out/'source_contract.json',d.read(d.SOURCE));torch.save(dict(model={k:v.cpu() for k,v in m.state_dict().items()},config=c,parent_epochs=20),out/'initialization.pt')
  losses=[];parent_NN=0
  with gzip.open(out/'step_metrics.jsonl.gz','wt',encoding='utf8') as log:
   for step in range(d.UPDATES):
    epoch=step//d.STEPS+1;lr=1e-6+.5*(.0002-1e-6)*(1+math.cos(math.pi*(epoch-1)/10));opt.param_groups[0]['lr']=lr
    ix=indices[step].tolist();x=L['x'][ix].to('cuda:0');y=L['y'][ix].long().to('cuda:0');second=tensor_iq(views[step],'cuda:0') if c['As'] else x
    if m.training or parent.training or any(p.requires_grad for p in parent.parameters()):raise ValueError('Common deterministic eval/autograd and frozen reference required')
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():t=parent.forward_identity_only(x)['tx_logits'];parent_NN+=128
    z=m.forward_identity_only(x)['tx_logits'];a=m.forward_identity_only(second)['tx_logits'];loss,metrics=objective(z,a,t,y);loss.backward()
    grads={n:float(p.grad.detach().square().sum()) for n,p in m.named_parameters() if p.requires_grad and p.grad is not None}
    if not grads or not all(math.isfinite(v) for v in grads.values()):raise ValueError('No finite feature gradient')
    norm=float(optim.clip(m,c,opt));opt.step()
    if not all(torch.equal(v,m.state_dict()[k]) for k,v in fixed.items()):raise ValueError('Frozen native F changed')
    if d.HEAD not in grads or not any(n!=d.HEAD for n in grads):raise ValueError('Default W and real features must participate in the same backward')
    record=dict(step=step+1,epoch=epoch,cumulative_epoch=20+epoch,L_indices=ix,As_consumed=bool(c['As']),bank_step=step+1,lr=lr,gradient_norm=norm,gradient_squared_by_parameter=grads,student_NN=256,parent_NN=128,head_frozen=False,head_joint=True,U_used=False,raw_supplement_NN=256,**metrics);log.write(json.dumps(record,allow_nan=False)+'\n');losses.append(metrics)
    if (step+1)%d.STEPS==0:
     log.flush();mean={k:float(np.mean([r[k] for r in losses])) for k in ('clean_CE','second_CE','keep_KL','total_loss')};losses=[]
     d.write(out/'reference_progress.json',dict(epoch=epoch,cumulative_epoch=20+epoch,step=step+1,**mean,seconds=time.perf_counter()-start));print(json.dumps(dict(epoch=epoch,cumulative_epoch=20+epoch,step=step+1,**mean)),flush=True)
  if m.id_backbone.encoder._full_eval_packets!=2220*256 or parent_NN!=2220*128:raise ValueError('Exact L-only forward budget differs')
  if m.id_backbone.encoder.core.id_backbone.raw_temporal.packets!=568320:raise ValueError('Supplemental encoder must run once per actual student packet')
  if torch.equal(m.state_dict()[d.HEAD],parent_state[d.HEAD]):raise ValueError('Default W did not jointly update')
  torch.save(dict(model={k:v.detach().cpu() for k,v in m.state_dict().items()},config=c,epoch=30,adaptation_epoch=10,optimizer_steps=2220,checkpoint_selection='fixed FINAL30',default_head_frozen=False,default_head_joint=True,optimizer=opt.state_dict()),out/'final_ssdg.pth')
  d.write(out/'source_frozen.json',dict(config=c,selection='fixed FINAL30',target_access=False,head_fit_updates=0,source_only=True,default_head_frozen=False,default_head_joint=True))
  d.write(out/'completion.json',dict(status='SOURCE_TRAINED',config=c,epoch=30,adaptation_epochs=10,parent_epochs=20,optimizer_steps=2220,student_training_NN=568320,parent_reference_NN=284160,initial_source_readout_NN=50400,source_only=True,target_access=False,head_fit_updates=0,head_joint_optimizer_steps=2220,raw_supplement_training_NN=568320,U_used=False,EMA_used=False,**counts,wall_seconds=time.perf_counter()-start))
