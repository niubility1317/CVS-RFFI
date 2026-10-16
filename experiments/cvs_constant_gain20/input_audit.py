"""Synthetic no-query probe of actual F inputs/joint/default head;no accuracy."""
from types import MethodType
import torch
from . import design as d
VIEWS=('identity','gain0.9','gain1.1','gain0.9_RMSrestored','gain1.1_RMSrestored')
def synthetic():
 g=torch.Generator().manual_seed(d.AUDIT_SEED);noise=torch.randn(32,2,256,generator=g);x=noise.clone()
 for t in range(1,256):x[8:16,:,t]=.7*x[8:16,:,t-1]+noise[8:16,:,t]
 x[16:24]=torch.nn.functional.avg_pool1d(noise[16:24].sign(),5,stride=1,padding=2)+.03*noise[16:24]
 n=torch.linspace(0,1,256);phase=2*torch.pi*(3*n+11*n.square());x[24:32]=torch.stack((phase.cos(),phase.sin())).unsqueeze(0)+.05*noise[24:32]
 return x/x.square().sum(1).mean(1).sqrt()[:,None,None]
def view(x,name):
 if name not in VIEWS or x.shape!=(32,2,256) or not bool(torch.isfinite(x).all()):raise ValueError('Fixed finite synthetic audit matrix only')
 gain=.9 if name.startswith('gain0.9') else 1.1 if name.startswith('gain1.1') else 1.
 y=x*gain
 if name.endswith('_RMSrestored'):y=y*(x.square().sum(1).mean(1)/y.square().sum(1).mean(1)).sqrt()[:,None,None]
 return y,gain
def measures(joint,logits,raw_joint,raw_logits):
 p=torch.nn.functional.normalize(joint.double(),dim=1,eps=1e-4);q=torch.nn.functional.normalize(raw_joint.double(),dim=1,eps=1e-4)
 rel=(p-q).norm(dim=1)/q.norm(dim=1).clamp_min(1e-20)
 return dict(normalized_joint_change=rel.tolist(),normalized_joint_change_median=float(rel.median()),normalized_joint_change_max=float(rel.max()),logits_abs_change_max=float((logits.double()-raw_logits.double()).abs().max()),raw_joint_relative_change=float((joint.double()-raw_joint.double()).norm()/raw_joint.double().norm().clamp_min(1e-20)))
def qualify(values):
 if set(values)!=set(VIEWS):raise ValueError('All fixed audit views required')
 controls=[values[n] for n in VIEWS if n.endswith('_RMSrestored')]
 conditions=dict(gain_reaches_actual_sinc=all(values[n]['gain_RMS_relative_error_max']<=1e-6 for n in VIEWS),restored_input=all(v['restored_input_relative_error_max']<=1e-6 for v in controls),restored_function=all(v['normalized_joint_change_max']<=1e-4 and v['logits_abs_change_max']<=1e-3 for v in controls),nontrivial_default_function=any(values[n]['normalized_joint_change_median']>=1e-3 and values[n]['logits_abs_change_max']>=.01 for n in ('gain0.9','gain1.1')))
 return dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions)
def run(m):
 ref=m.id_backbone.encoder;b=ref.core.id_backbone;raw=synthetic();before={k:v.detach().clone() for k,v in m.state_dict().items()};captured={};results={};traces={};original_features=ref.features;original_sinc=b.sinc.forward_iq_pair
 def features(self,x):
  captured['encoder_input']=x.detach().clone();value=original_features(x);captured['joint']=value.detach().clone();return value
 def sinc(self,x):captured['sinc_input']=x.detach().clone();return original_sinc(x)
 def save(name):
  def hook(module,args,out):captured[name]=out.detach().clone()
  return hook
 ref.features=MethodType(features,ref);b.sinc.forward_iq_pair=MethodType(sinc,b.sinc);handles=[b.time_down.register_forward_hook(save('time_down')),b.t2.pool.register_forward_hook(save('t2_pool')),ref.response.register_forward_hook(save('reference'))]
 try:
  with torch.no_grad():
   for name in VIEWS:
    captured.clear();x,gain=view(raw,name);z=m.forward_identity_only(x)['tx_logits'].double()
    if set(captured)!={'encoder_input','sinc_input','joint','time_down','t2_pool','reference'} or z.shape!=(32,6) or captured['joint'].shape!=(32,160) or any(not bool(torch.isfinite(v).all()) for v in captured.values()) or not bool(torch.isfinite(z).all()) or not torch.equal(x,captured['encoder_input']) or not torch.equal(x,captured['sinc_input']) or bool((captured['reference'][:,27]!=0).any()):raise ValueError('Actual fixed F/defaulthead/NoExplicitCFO input path differs')
    if name=='identity':base={k:v.clone() for k,v in captured.items()};raw_logits=z.clone()
    value=measures(captured['joint'],z,base['joint'],raw_logits);power=captured['sinc_input'].square().sum(1).mean(1);expected=raw.square().sum(1).mean(1)*(1. if name.endswith('_RMSrestored') else gain**2)
    value.update(gain_RMS_relative_error_max=float(((power-expected).abs()/expected).max()),restored_input_relative_error_max=float((x-raw).flatten(1).norm(dim=1).div(raw.flatten(1).norm(dim=1)).max()) if name.endswith('_RMSrestored') else None,stage_changes={k:dict(relative_L2=float((captured[k]-base[k]).double().norm()/base[k].double().norm().clamp_min(1e-20)),shape=list(captured[k].shape)) for k in ('time_down','t2_pool','reference')});results[name]=value
    traces[name]=dict(joint=captured['joint'].tolist(),logits=z.tolist(),encoder_input_complex_power=captured['encoder_input'].square().sum(1).mean(1).tolist(),sinc_input_complex_power=power.tolist())
 finally:
  ref.features=original_features;b.sinc.forward_iq_pair=original_sinc
  for h in handles:h.remove()
 if any(not torch.equal(before[k],v) for k,v in m.state_dict().items()) or any(p.grad is not None for p in m.parameters()) or m.training or m._joint_calls or m._joint_As_packets or m._gain_packets or m._supcon_calls or m.id_backbone.encoder._view_eval_packets!=162 or m.id_backbone.encoder._view_train_packets:raise ValueError('Frozen state/actual NN cost differs')
 d.write(d.BASE/'input_audit_traces.json',traces);result=dict(status='COMPLETE_SYNTHETIC_NO_QUERY',qualification=qualify(results),views=results,synthetic_audit_NN_packets=160,actual_total_NN_packets=162,real_source_IQ_reads=0,training_updates=0,head_fit_updates=0,target_access=False,identity_accuracy_computed=False,model_state_unchanged=True)
 d.write(d.BASE/'input_audit.json',result);return result
