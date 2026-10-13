import copy,json
from types import SimpleNamespace
import torch
from . import design as d
from .gates import FIELDS,physical_metadata,masks,expected_neighbor_visibility
from experiments.cvs_phase1_overlay.model import native_modules
from experiments.cvs_feature_multiscale50.model import build

def checks():
 torch.set_num_threads(2);native=native_modules();from cvsrffi.game_tracking.data import opaque_id
 y=[];rx=[];day=[];ids=[];base=[]
 for tx in range(6):
  for ri,r in enumerate([1,3,4,6,8]):
   for di,t in enumerate([1,2,3]):
    for sig in range(70):
     y.append(tx);rx.append(r);day.append(t);ids.append(opaque_id(SimpleNamespace(tx_i=tx,rx_i=r,day_i=t,eq_i=1,sig_i=sig)));base.append(tx*15000+ri*3000+di*1000+sig)
 data=dict(y=torch.tensor(y),rx=torch.tensor(rx),day=torch.tensor(day),ids=ids)
 meta=physical_metadata(data,'L_s');assert set(meta)==FIELDS and torch.equal(meta['base_index'],torch.tensor(base))
 negatives=0
 for role,bad in [('U_s',data),('L_s',dict(data,y=torch.full((6300,),-1)))]:
  try:physical_metadata(bad,role)
  except ValueError:negatives+=1
  else:raise AssertionError('Hidden U/source role accepted')
 args=d.old.make_args(d.source_config(d.rows()[0]),'cpu')
 meta=dict(rx_i=torch.tensor([1,1,3,3]),day_i=torch.ones(4,dtype=torch.long),eq_i=torch.ones(4,dtype=torch.long),sig_i=torch.tensor([100,101,100,104]),base_index=torch.tensor([100,101,3100,3104]))
 weak=torch.zeros(4,6);weak[0:2,0]=12;weak[2:4,1]=12;strong=weak.clone();strong[1].zero_();strong[1,1]=12
 pseudo,selected,detail=masks(native,weak,strong,meta,torch.tensor([0,0,3,3]),args)
 assert selected['CONFIDENCE'].sum()==4 and selected['CONFIDENCE_STRONG'].sum()==3 and selected['FULL_OLD'].sum()==1 and selected['GEOMETRIC_NEIGHBOR'].sum()==2
 try:masks(native,weak,strong,dict(meta,tx_i=torch.zeros(4)),torch.zeros(4),args)
 except ValueError:negatives+=1
 else:raise AssertionError('TX truth metadata accepted by gates')
 # Exact hypergeometric no-neighbor probability on a tiny publicly synthetic pool.
 value=expected_neighbor_visibility(2560,70)['expected_neighbor_visibility'];assert 0<value<1
 try:expected_neighbor_visibility(6300,70,window=3)
 except ValueError:negatives+=1
 else:raise AssertionError('Unregistered window accepted')
 torch.manual_seed(42);model=build(d.source_config(d.rows()[0]),'cpu').eval().requires_grad_(False);x=torch.randn(4,2,256)
 state={k:v.clone() for k,v in model.state_dict().items()}
 with torch.no_grad():
  full=model(x,y_tx=None,grl_lambda=1.,return_aux=True,domain_labels=torch.zeros(4,dtype=torch.long))
  identity=model.forward_identity_only(x)
  assert torch.equal(full['tx_logits'],identity['tx_logits'])
  model.train()
  assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.id_backbone.modules())
  model.forward_identity_only(native._strong_augment(x,.015))
 model.eval()
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
 for change in [dict(roles=['U_s']),dict(target_access=True),dict(optimizer_steps=1),dict(head='new classifier')]:
  c=copy.deepcopy(d.config(d.rows()[0]));c.update(change)
  try:d.validate(c)
  except ValueError:negatives+=1
  else:raise AssertionError('Unregistered source/head fitting accepted')
 return dict(status='PASS',synthetic_only=True,no_real_IQ_or_checkpoint_or_truth_read=True,native_physical_metadata_mapping_checked=True,label_keys_rejected=True,native_gate_masks_expected=True,eval_identity_logits_equal_native_U_path=True,strong_train_identity_state_unchanged=True,negative_checks=negatives,head_updates=0,optimizer_steps=0)
if __name__=='__main__':print(json.dumps(checks()))
