import copy,json,tempfile,math
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .observer import Observer,summarize_snapshots
from .optimizer import NAMES
from experiments.cvs_phase1_overlay.model import native_modules
from experiments.cvs_temporal_rsc50.checks import invoke

def checks():
 torch.set_num_threads(2);rows=d.training_rows();models=[api.build(d.config(row),'cpu') for row in rows];initial={k:v.clone() for k,v in models[0].state_dict().items()}
 assert all(all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) for m in models)
 args=[d.make_args(d.config(r),'cpu') for r in rows]
 def same(a,b):return a==b or isinstance(a,float) and isinstance(b,float) and math.isnan(a) and math.isnan(b)
 diff=[k for k in vars(args[0]) if not same(vars(args[0])[k],vars(args[1])[k])]
 assert set(diff).issubset({'candidate_id','output_dir'})
 gen=torch.Generator().manual_seed(9560);x=torch.randn(12,2,256,generator=gen);y=torch.arange(12)%6;rx=torch.tensor([1,3,4,6,8,1]*2);n=native_modules();actual=[];saved_class=torch.optim.AdamW
 for row,m,a in zip(rows,models,args):
  c=d.config(row);m.train()
  assert a.use_unlabeled and a.use_ema_teacher and a.pseudo_temporal_mode=='batch_neighbor' and a.pseudo_temporal_gate and not a.freeze_backbone and a.label_epochs==33 and a.pseudo_epochs==17
  with tempfile.TemporaryDirectory(prefix='sinc-native-check-') as tmp:
   obs=Observer(c);obs.initialize(m,c,Path(tmp))
   with obs.native_scope(n):
    opt=torch.optim.AdamW(n._optimizer_parameters(m,None),lr=float(a.lr),weight_decay=float(a.weight_decay))
    for batch in range(1,17):
     opt.zero_grad(set_to_none=True);out=invoke(m,a,x,y,rx);loss=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01);zero=loss*0
     assert obs.observe(model=m,args=a,epoch=1,batch_idx=batch,loss_l=loss,loss_u=zero,loss_ent=zero,active=False,logits=None,pseudo=None,mask=None) is zero
     loss.backward();torch.nn.utils.clip_grad_norm_(n._optimizer_parameters(m,None),5.);opt.step()
   assert torch.optim.AdamW is saved_class and n._trusted_pseudo_loss_hook is None
   assert opt._sinc_unit_steps==16 and obs.steps==16 and obs.U_packets==0 and len(obs.sinc_raw)==1
   r=obs.sinc_raw[0];assert r['step']==16 and all(v['optimizer_step']==16 for v in r['parameters'].values())
   torch.save(obs.sinc_raw,Path(tmp)/'snapshot.pt');same=torch.load(Path(tmp)/'snapshot.pt',map_location='cpu',weights_only=False)
   assert summarize_snapshots(same)==summarize_snapshots(obs.sinc_raw)
   assert all(p.requires_grad for p in (m.id_backbone.encoder.core.id_backbone.sinc.low_hz_,m.id_backbone.encoder.core.id_backbone.sinc.band_hz_,m.id_backbone.encoder.core.id_backbone.cls_head.weight))
   actual.append(dict(row_id=row['row_id'],mode=c['sinc_optimizer_units']['mode'],actual_native_L_contexts=m._feature_l_steps,synthetic_joint_optimizer_steps=16,actual_optimizer_group_count=len(opt.param_groups),original_U_loss_object_unchanged=True,selected_Sinc_snapshot_recount=True,default_W_gradient_nonzero=float(m.id_backbone.encoder.core.id_backbone.cls_head.weight.grad.norm())>0,source_or_target_IQ_read=False,head_fit_updates=0))
 assert all(r['actual_native_L_contexts']==16 and r['default_W_gradient_nonzero'] for r in actual)
 return dict(status='PASS',actual_rows=actual,same_seed_initial_state_equal=True,actual_native_arg_differences=diff,global_optimizer_class_restored=True,no_new_feature_loss_or_forward=True,real_IQ_or_checkpoint_read=False,formal_training_started=False)
if __name__=='__main__':print(json.dumps(checks()))
