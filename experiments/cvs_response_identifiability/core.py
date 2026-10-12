import torch
from experiments.cvs_finite_action.predictors import inputs

def origin_prediction(reader,h,q,a):
 # Reader outputs standardized delta; undo mean before differencing physical responses.
 return reader(h,q,a)-reader(h,q,torch.zeros_like(a))-reader.state['y']['mean']/reader.state['y']['scale']

def jacobian(reader,h,q,a,seed,probes):
 hn=((h-reader.state['h']['mean'])/reader.state['h']['scale']).detach().requires_grad_(True)
 qn=((q-reader.state['q']['mean'])/reader.state['q']['scale']).detach().requires_grad_(True)
 hr=hn*reader.state['h']['scale']+reader.state['h']['mean'];qr=qn*reader.state['q']['scale']+reader.state['q']['mean']
 direct=reader(hr,qr,a);origin=origin_prediction(reader,hr,qr,a);gen=torch.Generator().manual_seed(seed);result={}
 for mode,prediction in [('original',direct),('zero_origin',origin)]:
  hs=qs=0.
  for probe in range(probes):
   v=(torch.randint(0,2,prediction.shape,generator=gen).to(prediction.device,dtype=prediction.dtype)*2-1)
   gh,gq=torch.autograd.grad((prediction*v).sum(),(hn,qn),retain_graph=True)
   hs+=float(gh.square().sum(1).mean());qs+=float(gq.square().sum(1).mean())
  hs/=probes;qs/=probes
  result[mode]=dict(h_frobenius_squared=hs,q_frobenius_squared=qs,h_entry_RMS=(hs/(hn.shape[1]*prediction.shape[1]))**.5,q_entry_RMS=(qs/(qn.shape[1]*prediction.shape[1]))**.5,output_dimensions=prediction.shape[1],h_dimensions=hn.shape[1],q_dimensions=qn.shape[1],samples=len(h),probes=probes)
 if any(p.grad is not None for p in reader.parameters()):raise ValueError('Frozen D gradients changed')
 return result

@torch.no_grad()
def errors(reader,parameter,h,h0,q,a,actual,target0,shuffle,device,batch=512):
 values={};n=len(h);sc=reader.state;hmean=sc['h']['mean'];qmean=sc['q']['mean']
 for start in range(0,n,batch):
  sl=slice(start,start+batch);hh=h[sl].to(device);base=h0[sl].to(device);qq=q[sl].to(device);aa=a.expand(len(hh),-1)
  y0=(target0[sl].to(device)-sc['y']['mean'])/sc['y']['scale'];yt=(actual[sl].to(device)-sc['y']['mean'])/sc['y']['scale']
  pred=reader(hh,qq,aa);zero=reader(hh,qq,torch.zeros_like(aa));corrected=pred-zero-sc['y']['mean']/sc['y']['scale']
  candidates=dict(reference_target=pred,zero_origin_target=corrected,fit_h_mean=reader(hmean.expand_as(hh),qq,aa),fit_q_mean=reader(hh,qmean.expand_as(qq),aa),stratum_shuffle_h=reader(h[shuffle[sl]].to(device),qq,aa),stratum_shuffle_q=reader(hh,q[shuffle[sl]].to(device),aa),original_parameter_linear=inputs(hh,qq,aa,sc,parameter=True)@parameter)
  metrics={k:(v-y0).square().mean(1) for k,v in candidates.items()}
  metrics.update(actual_closure=(pred-yt).square().mean(1),zero_origin_actual_closure=(corrected-yt).square().mean(1),actual_response_drift=((actual[sl].to(device)-target0[sl].to(device))/sc['y']['scale']).square().mean(1),prediction_drift=(pred-reader(base,qq,aa)).square().mean(1),clean_feature_drift=((hh-base)/sc['h']['scale']).square().mean(1),zero_parameter_output_energy=(zero+sc['y']['mean']/sc['y']['scale']).square().mean(1),reference_response_energy=(target0[sl].to(device)/sc['y']['scale']).square().mean(1),actual_response_energy=(actual[sl].to(device)/sc['y']['scale']).square().mean(1))
  if any(not torch.isfinite(v).all() for v in metrics.values()):raise FloatingPointError('Nonfinite response diagnostic')
  for key,value in metrics.items():values.setdefault(key,[]).append(value.cpu())
 return {k:torch.cat(v) for k,v in values.items()}
