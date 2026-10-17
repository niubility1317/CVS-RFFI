"""Real F architecture,synthetic IQ only;no checkpoint or optimization."""
import json,sys
from pathlib import Path
import torch
from .iq_direction import make_direction,permuted_control,complex_rms,EPS,PEAK

def run():
 root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'experiments/adv3b02_xuc/code'),str(root/'code')]
 from SSDG import train_ssdg as native
 from experiments.cvs_scene_cycle20 import design as d,model as api
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 torch.set_num_threads(2);torch.manual_seed(202610178103);m=api.build(d.config(d.rows()[0]),'cpu').eval();m.requires_grad_(False)
 x=torch.randn(4,2,256,generator=torch.Generator().manual_seed(202610178104));x=x/complex_rms(x).float().view(-1,1,1);saved=x.clone();before={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone()
 fn=lambda value:m.forward_identity_only(value)['tx_logits'];cache=lambda:fresh_parameter_caches(m)
 delta,e=make_direction(m,[x],fn,202610178001,cache_context=cache);control=permuted_control(delta,202610178002)
 assert e['identifiable'] and e['reference_packets']==4 and e['eval_forward_packets']==8 and e['input_backward_packets']==4
 assert float(complex_rms(delta)[0])<=EPS*(1+1e-6) and float(delta.abs().max())<=PEAK*(1+1e-6) and not delta.requires_grad and torch.equal(x,saved)
 predictions=[]
 for v in (x,x+delta,x+control):
  with torch.no_grad(),cache():logits=fn(v)
  assert logits.shape==(4,6) and bool(torch.isfinite(logits).all());predictions.append(logits)
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and all(p.grad is None for p in m.parameters()) and torch.equal(rng,torch.get_rng_state())
 assert not torch.equal(predictions[0],predictions[1]) and not torch.equal(predictions[1],predictions[2])
 assert m.id_backbone.encoder._view_train_packets==0 and m.id_backbone.encoder._view_eval_packets==20
 return dict(status='PASS',architecture='F+CROSS_RX/X3/unifiedSinc/joint160/defaultcosine',source_only_probe_not_training_recipe=True,synthetic_forward_packets=20,synthetic_input_backward_packets=4,optimizer_updates=0,real_IQ=0,checkpoint_loads=0,target_access=False,direction_RMS=e['direction_RMS'],direction_peak=e['direction_peak'],same_energy_control=True,input_gradient_exists=True,default_head_outputs_change=True,model_and_RNG_unchanged=True,scope='Engineering only;no RF identity or domain-generalization evidence')
if __name__=='__main__':print(json.dumps(run()))
