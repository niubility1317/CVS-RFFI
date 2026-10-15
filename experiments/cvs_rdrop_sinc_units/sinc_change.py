"""Actual initial/final checkpoint Sinc movement, without any IQ or fitting.

The caller verifies source provenance before loading checkpoints. Synthetic
checks below validate arithmetic only; real measurements require saved weights.
"""
import copy,json,torch
PREFIX='id_backbone.encoder.core.id_backbone.sinc.'

def compare(initial,final,SincType):
 keys=['low_hz_','band_hz_','t_','window_']
 if any(PREFIX+k not in s for s in (initial,final) for k in keys):raise ValueError('Missing actual identity Sinc state')
 values=[{k:s[PREFIX+k].detach().cpu() for k in keys} for s in (initial,final)]
 if any(v['low_hz_'].shape!=(24,1) or v['band_hz_'].shape!=(24,1) or v['t_'].shape!=(1,79) or v['window_'].shape!=(1,79) or any(not torch.isfinite(x).all() for x in v.values()) for v in values):raise ValueError('Unexpected Sinc dimensions or nonfinite weights')
 if any(not torch.equal(values[0][k],values[1][k]) for k in ('t_','window_')):raise ValueError('Different fixed sample-rate/window buffers')
 modules=[]
 for v in values:
  m=SincType(24,79,sample_rate=25e6,dataset='wisig',input_len=256).eval().requires_grad_(False)
  m.load_state_dict(v,strict=True);modules.append(m)
 with torch.no_grad():kernels=[m._filters(torch.device('cpu'),torch.float32).clone() for m in modules]
 def describe(v):
  low=(50+v['low_hz_'].abs()).clamp(50,12.5e6-51);high=torch.clamp(low+50+v['band_hz_'].abs(),min=low+50,max=torch.full_like(low,12.5e6-1))
  return low,high
 edges=[describe(v) for v in values];delta=kernels[1]-kernels[0]
 stats={}
 for name in ('low_hz_','band_hz_'):
  change=values[1][name]-values[0][name];ulp=torch.nextafter(values[0][name],torch.full_like(values[0][name],float('inf')))-values[0][name]
  stats[name]=dict(elements=change.numel(),exactly_unchanged=int((change==0).sum()),abs_change_Hz_mean=float(change.abs().mean()),abs_change_Hz_max=float(change.abs().max()),relative_L2=float(change.norm()/values[0][name].norm().clamp_min(1e-12)),initial_float32_ULP_Hz_min=float(ulp.min()),initial_float32_ULP_Hz_max=float(ulp.max()),channel_changes_Hz=change.flatten().tolist())
 low0,high0=edges[0];low1,high1=edges[1]
 rel=(delta.flatten(1).norm(dim=1)/kernels[0].flatten(1).norm(dim=1).clamp_min(1e-12))
 return dict(status='VERIFIED',actual_identity_prefix=PREFIX,sample_rate_Hz=25e6,channels=24,kernel_length=79,parameter_changes=stats,kernel_relative_L2=float(delta.norm()/kernels[0].norm().clamp_min(1e-12)),kernel_max_abs_change=float(delta.abs().max()),kernel_exactly_unchanged_channels=int((delta.abs().flatten(1).amax(1)==0).sum()),per_channel_kernel_relative_L2=rel.tolist(),effective_low_abs_change_Hz_max=float((low1-low0).abs().max()),effective_high_abs_change_Hz_max=float((high1-high0).abs().max()),initial_coverage_Hz=[float(low0.min()),float(high0.max())],final_coverage_Hz=[float(low1.min()),float(high1.max())],parameters_or_buffers_updated_by_diagnostic=0,IQ_read=False,query_read=False,inference='actual checkpoint movement; does not by itself prove Sinc usefulness,causal bottleneck or transfer benefit')
