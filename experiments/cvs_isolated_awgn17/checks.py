"""Focused native-noise equivalence,properness,protocol negatives and identity counterexamples."""
import copy,json,random,sys
import torch
from . import design as d
from .noise import generate
from .decision import qualify

def run():
 torch.set_num_threads(2);sys.path.insert(0,str(d.ROOT/'experiments/adv3b02_xuc/code'));import DataAugmentation as native
 x=torch.randn(32,2,256,generator=torch.Generator().manual_seed(202610172201));saved=x.clone();random.seed(202610172202);torch.manual_seed(202610172202);py=random.getstate();ts=torch.random.get_rng_state();snr=native._rand_uniform((32,1),x.device,low=20.,high=36.,dtype=x.dtype);canonical=native._from_complex(native._apply_awgn(native._to_complex(x.clone()),snr));end=torch.random.get_rng_state();random.setstate(py);torch.random.set_rng_state(ts);out,stats=generate(x,native);assert torch.equal(canonical,out) and torch.equal(end,torch.random.get_rng_state()) and py==random.getstate() and torch.equal(x,saved) and stats['nominal_SNR_dB']==snr.view(-1).tolist();generations=64
 torch.manual_seed(202610172203);state=torch.random.get_rng_state();a,sa=generate(x,native);torch.random.set_rng_state(state);b,sb=generate(1.5*x,native);assert torch.allclose(b,1.5*a,rtol=1e-5,atol=5e-7) and max(abs(g-h) for g,h in zip(sa['relative_IQ_RMS_change'],sb['relative_IQ_RMS_change']))<1e-6;generations+=64
 large=torch.randn(1024,2,256,generator=torch.Generator().manual_seed(202610172204));torch.manual_seed(202610172205);v,s=generate(large,native);n=(v-large).double();power=large.double().square().sum(1).mean(1);snr=torch.tensor(s['nominal_SNR_dB'],dtype=torch.float64);sigma=torch.sqrt(power/(10**(snr/10))/2);u=n/sigma[:,None,None];z=torch.complex(u[:,0],u[:,1]);ratio=float((z*z).mean().abs()/z.abs().square().mean());lag=float((z[:,1:]*z[:,:-1].conj()).mean().abs()/z.abs().square().mean());assert float(u[:,0].mean().abs())<.01 and float(u[:,1].mean().abs())<.01 and abs(float(u[:,0].square().mean())-1)<.02 and abs(float(u[:,1].square().mean())-1)<.02 and ratio<.01 and lag<.01;generations+=1024
 negatives=0;bad=[None,x[:0],x[:,0],x[:,:,:255],x.double(),x.to(torch.int64),torch.zeros_like(x),torch.full_like(x,float('nan')),torch.full_like(x,float('inf')),x.clone().requires_grad_()]
 for value in bad:
  ts=torch.random.get_rng_state();py=random.getstate()
  try:generate(value,native)
  except (ValueError,TypeError):negatives+=1
  else:raise AssertionError('Illegal source input accepted')
  assert torch.equal(ts,torch.random.get_rng_state()) and py==random.getstate()
 for change in ({'target_access':True},{'nominal_SNR_dB':[30.,45.]},{'optimizer_updates':1},{'retain_all_tail_samples':False}):
  c=d.config();c.update(change)
  try:d.validate(c)
  except ValueError:negatives+=1
  else:raise AssertionError('Protocol override accepted')
 raw=dict(accuracy=.998,macro_F1=.998,**{'by_'+k:{'0':dict(accuracy=.998)} for k in ('RX','TX','day')});noise=dict(accuracy=.997,macro_F1=.997,raw_correct_retention=.995,**{'by_'+k:{'0':dict(accuracy=.997)} for k in ('RX','TX','day')});draws=[copy.deepcopy(noise) for _ in range(4)];st=dict(relative_IQ_RMS_change=[.02,.03,.05]);assert qualify(raw,noise,draws,st)['decision']=='PASS';counter=0
 for key,value in [('accuracy',.99),('macro_F1',.99),('raw_correct_retention',.98)]:
  z=copy.deepcopy(noise);z[key]=value;assert qualify(raw,z,draws,st)['decision']=='FAIL';counter+=1
 for k in ('RX','TX','day'):
  z=copy.deepcopy(noise);z['by_'+k]['0']['accuracy']=.98;assert qualify(raw,z,draws,st)['decision']=='FAIL';counter+=1
 dd=copy.deepcopy(draws);dd[2]['accuracy']=.98;assert qualify(raw,noise,dd,st)['decision']=='FAIL';counter+=1
 for changed in ([.001,.002,.003],[.02,.03,.16]):assert qualify(raw,noise,draws,dict(relative_IQ_RMS_change=changed))['decision']=='FAIL';counter+=1
 result=dict(status='PASS',input_config_negatives=negatives,scientific_counterexamples=counter,synthetic_native_noise_generations=generations,canonical_output_RNG=True,positive_gain_equivariance=True,Gaussian_IQ_moments=True,synthetic_pseudocov_ratio=ratio,synthetic_lag1_ratio=lag,source_IQ_reads=0,checkpoint_loads=0,NN=0,updates=0,all_input_and_Python_RNG_unchanged=True);print(json.dumps(result));return result
if __name__=='__main__':run()
