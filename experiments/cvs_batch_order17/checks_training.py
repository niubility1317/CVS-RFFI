"""Fixed config/protocol/scientific counterexamples and exact unchanged F checks."""
from copy import deepcopy
import json
import torch
from . import design as d,model as api
from experiments.cvs_scene_cycle20 import model as original

def healthy(arm):
 return dict(actual_steps=4440,L_forward_packets=4546560,extra_L_forward_packets=1136640,total_pair_weight=1.25,As_training_observations=0,head_fit_updates=0,target_access=False,final_base_L_objective=.3,order_epochs=20,order_steps=4440,order_blocks=740,moved_L_positions=0 if arm=='O0' else 560000,all_block_multisets_preserved=True,views={v:dict(accuracy=.8+(.02 if arm=='O1' else 0),macro_F1=.79+(.02 if arm=='O1' else 0)) for v in ('clean',*d.SCENES)},seen_V=dict(source_val_accuracy=.98))
def main():
 torch.set_num_threads(2);count=0
 for row in d.rows():
  c=d.config(row);d.validate(c)
  for mutate in (lambda z:z.update(epochs=21),lambda z:z.update(model_seed=2026092702),lambda z:z.update(checkpoint_sources=['forbidden']),lambda z:z.update(target_access=True),lambda z:z['batch_order'].update(block_updates=3),lambda z:z['batch_order'].update(U='changed'),lambda z:z['training_mechanism'].update(As=True),lambda z:z['training_mechanism'].update(raw_weight=.25)):
   z=deepcopy(c);mutate(z)
   try:d.validate(z)
   except ValueError:count+=1
   else:raise AssertionError('Unregistered config accepted')
 good={a:healthy(a) for a in d.ARMS};assert d.qualify(good)['qualified']
 for field,value in (('actual_steps',4439),('order_blocks',739),('head_fit_updates',1),('target_access',True),('all_block_multisets_preserved',False),('As_training_observations',128)):
  bad=deepcopy(good);bad['O1'][field]=value;assert not d.qualify(bad)['qualified'];count+=1
 for view,key,val in (('clean','accuracy',.809),('clean','macro_F1',.78),(d.SCENES[0],'accuracy',.79),(d.SCENES[-1],'accuracy',.79)):
  bad=deepcopy(good);bad['O1']['views'][view][key]=val;assert not d.qualify(bad)['qualified'];count+=1
 torch.manual_seed(d.SEED);base=original.build(d.legacy_config(),'cpu').eval();states=[];outputs=[];g=torch.Generator().manual_seed(202610175399);x=torch.randn(2,2,256,generator=g)
 with torch.no_grad():base_z=base.forward_identity_only(x)['tx_logits']
 state=base.state_dict();W='id_backbone.encoder.core.id_backbone.cls_head.weight';assert state[W].shape==(6,160)
 for row in d.rows():
  torch.manual_seed(d.SEED);m=api.build(d.config(row),'cpu').eval();s=m.state_dict();assert set(s)==set(state) and all(torch.equal(s[k],v) for k,v in state.items())
  assert not m._supcon_handles and m._supcon_pending is None and m._joint_calls==0 and m._joint_As_packets==0
  with torch.no_grad():z=m.forward_identity_only(x)['tx_logits']
  assert z.shape==(2,6) and torch.equal(base_z,z);states.append(len(s));outputs.append(z.shape[0])
 result=dict(status='PASS',protocol_and_scientific_counterexamples=count,exact_original_F_state_tensors=len(state),default_head_shape=list(state[W].shape),both_models_logits_bit_exact_to_original=True,new_parameters=0,new_IQ_transform=False,synthetic_NN_packets=6,optimizer_updates=0,real_IQ=0,checkpoint_loads=0,target_access=False)
 print(json.dumps(result))
if __name__=='__main__':main()
