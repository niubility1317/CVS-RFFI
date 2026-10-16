"""Actual whole F/default head and reference compensation;synthetic data only."""
from types import MethodType
import torch
from . import design as d
from .operator import view
def synthetic():
 g=torch.Generator().manual_seed(d.AUDIT_SEED);noise=torch.randn(32,2,256,generator=g);x=noise.clone()
 for t in range(1,256):x[8:16,:,t]=.7*x[8:16,:,t-1]+noise[8:16,:,t]
 x[16:24]=torch.nn.functional.avg_pool1d(noise[16:24].sign(),5,stride=1,padding=2)+.03*noise[16:24]
 n=torch.linspace(0,1,256);phase=2*torch.pi*(3*n+11*n.square());x[24:32]=torch.stack((phase.cos(),phase.sin())).unsqueeze(0)+.05*noise[24:32]
 return x/x.square().sum(1).mean(1).sqrt()[:,None,None]
def measures(joint,logits,raw_joint,raw_logits):
 p=torch.nn.functional.normalize(joint.double(),dim=1,eps=1e-4);q=torch.nn.functional.normalize(raw_joint.double(),dim=1,eps=1e-4)
 rel=(p-q).norm(dim=1)/q.norm(dim=1).clamp_min(1e-20)
 return dict(normalized_joint_change=rel.tolist(),normalized_joint_change_median=float(rel.median()),normalized_joint_change_max=float(rel.max()),logits_abs_change_max=float((logits.double()-raw_logits.double()).abs().max()))
def qualify(values):
 if set(values)!=set(d.VIEWS):raise ValueError('All five fixed frequency views required')
 controls=[values[n] for n in ('positive_inverse','negative_inverse')]
 conditions=dict(power_preserved=all(v['power_relative_error_max']<=1e-6 for v in values.values()),inverse_input=all(v['inverse_input_relative_error_max']<=1e-6 for v in controls),inverse_function=all(v['normalized_joint_change_max']<=1e-4 and v['logits_abs_change_max']<=1e-3 for v in controls),nontrivial_default_function=any(values[n]['normalized_joint_change_median']>=1e-3 and values[n]['logits_abs_change_max']>=.01 for n in ('frequency_positive','frequency_negative')))
 return dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions)
def run(m):
 ref=m.id_backbone.encoder;b=ref.core.id_backbone;raw=synthetic();before={k:v.detach().clone() for k,v in m.state_dict().items()};captured={};results={};traces={};original_features=ref.features;original_sinc=b.sinc.forward_iq_pair
 def features(self,x):captured['encoder_input']=x.detach().clone();v=original_features(x);captured['joint']=v.detach().clone();return v
 def sinc(self,x):captured['sinc_input']=x.detach().clone();return original_sinc(x)
 def save(name):
  def hook(module,args,out):captured[name]=out.detach().clone()
  return hook
 ref.features=MethodType(features,ref);b.sinc.forward_iq_pair=MethodType(sinc,b.sinc);handles=[b.time_down.register_forward_hook(save('time_down')),b.t2.pool.register_forward_hook(save('t2_pool')),ref.response.register_forward_hook(save('reference'))]
 try:
  with torch.no_grad():
   for name in d.VIEWS:
    captured.clear();x=view(raw,name);z=m.forward_identity_only(x)['tx_logits'].double()
    if set(captured)!={'encoder_input','sinc_input','joint','time_down','t2_pool','reference'} or z.shape!=(32,6) or captured['joint'].shape!=(32,160) or any(not bool(torch.isfinite(v).all()) for v in captured.values()) or not bool(torch.isfinite(z).all()) or not torch.equal(x,captured['encoder_input']) or not torch.equal(x,captured['sinc_input']) or bool((captured['reference'][:,27]!=0).any()):raise ValueError('Actual F/defaulthead/NoExplicitCFO input differs')
    if name=='identity':base={k:v.clone() for k,v in captured.items()};raw_logits=z.clone()
    value=measures(captured['joint'],z,base['joint'],raw_logits);power=captured['sinc_input'].square().sum(1).mean(1);expected=raw.square().sum(1).mean(1)
    value.update(power_relative_error_max=float(((power-expected).abs()/expected).max()),inverse_input_relative_error_max=float((x-raw).flatten(1).norm(dim=1).div(raw.flatten(1).norm(dim=1)).max()) if name.endswith('_inverse') else None,stage_changes={k:dict(relative_L2=float((captured[k]-base[k]).double().norm()/base[k].double().norm().clamp_min(1e-20)),shape=list(captured[k].shape)) for k in ('time_down','t2_pool','reference')});results[name]=value
    traces[name]=dict(joint=captured['joint'].tolist(),logits=z.tolist(),encoder_input_complex_power=captured['encoder_input'].square().sum(1).mean(1).tolist(),sinc_input_complex_power=power.tolist())
 finally:
  ref.features=original_features;b.sinc.forward_iq_pair=original_sinc
  for h in handles:h.remove()
 if any(not torch.equal(before[k],v) for k,v in m.state_dict().items()) or any(p.grad is not None for p in m.parameters()) or m.training or m._joint_calls or m._joint_As_packets or m._supcon_calls or ref._view_eval_packets!=162 or ref._view_train_packets:raise ValueError('Frozen state/actual NN cost differs')
 d.write(d.BASE/'input_audit_traces.json',traces);result=dict(status='COMPLETE_SYNTHETIC_NO_QUERY',qualification=qualify(results),views=results,synthetic_audit_NN_packets=160,actual_total_NN_packets=162,real_source_IQ_reads=0,training_updates=0,head_fit_updates=0,target_access=False,identity_accuracy_computed=False,model_state_unchanged=True,reference_compensation='Unmodified NoExplicitCFO ExcitationResponse;mask coord27 only,not whole-model synchronizer')
 d.write(d.BASE/'input_audit.json',result);return result
