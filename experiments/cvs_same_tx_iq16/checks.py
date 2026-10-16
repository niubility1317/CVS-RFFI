"""Focused transform, label/role pairing and scientific-counterexample checks."""
from copy import deepcopy
import random
import torch
from .iqmix import mix as transform,donor_indices,qualify
from . import design as d

def run():
 operator_packets=0;zero_coefficient_packets=0
 def mix(*args,**kwargs):
  nonlocal operator_packets,zero_coefficient_packets
  result=transform(*args,**kwargs);co=kwargs.get('coefficient',args[2] if len(args)>2 else .1)
  if co:operator_packets+=len(args[0])
  else:zero_coefficient_packets+=len(args[0])
  return result
 torch.set_num_threads(2);g=torch.Generator().manual_seed(20261016171);x=torch.randn(16,2,256,generator=g);y=torch.randn(16,2,256,generator=g);before=x.clone();donor_before=y.clone();rng=torch.random.get_rng_state();py=random.getstate()
 a,info=mix(x,y);assert torch.equal(x,before) and torch.equal(y,donor_before) and torch.equal(rng,torch.random.get_rng_state()) and py==random.getstate()
 assert max(info['relative_IQ_change'])<=.25001 and max(info['relative_power_error'])<=1e-5 and torch.equal(mix(x,y,0.)[0],x)
 assert torch.allclose(mix(x,x)[0],x,atol=1e-6,rtol=1e-6) and torch.equal(mix(torch.zeros_like(x),y)[0],torch.zeros_like(x))
 assert torch.allclose(mix(x,torch.zeros_like(y))[0],x,atol=1e-6,rtol=1e-6)
 n=torch.arange(256,dtype=torch.float64);tone=torch.exp(2j*torch.pi*2*n/20)[None].repeat(8,1);impaired=3*tone*torch.exp(1j*(.4+.003*n))
 convert=lambda z:torch.stack((z.real,z.imag),1).float()
 restored=mix(convert(tone),convert(impaired))[0];assert torch.allclose(restored,convert(tone),atol=1e-6,rtol=1e-6)
 az=torch.complex(a[:,0].double(),a[:,1].double());xz=torch.complex(x[:,0].double(),x[:,1].double());yz=torch.complex(y[:,0].double(),y[:,1].double());rotation=complex(.8,.6)
 rotated=mix(convert(xz*rotation*1.3),convert(yz*rotation*1.3))[0];expected=convert(az*rotation*1.3)
 common_error=float(((rotated.double()-expected.double()).flatten(1).norm(dim=1)/expected.double().flatten(1).norm(dim=1)).max());assert common_error<=8*torch.finfo(torch.float32).eps
 donor_rotated=mix(x,convert(yz*rotation*2.))[0];donor_error=float(((donor_rotated.double()-a.double()).flatten(1).norm(dim=1)/a.double().flatten(1).norm(dim=1)).max());assert donor_error<=8*torch.finfo(torch.float32).eps
 labels=[];rx=[];day=[];ids=[]
 for yy in range(6):
  for rr in d.FIT:
   for dd in (1,2,3):
    for k in range(70):labels.append(yy);rx.append(rr);day.append(dd);ids.append(f'opaque-{yy}-{rr}-{dd}-{k:02d}')
 same,cross=donor_indices(labels,rx,day,ids)
 pairs=lambda same,cross,identities:{identities[i]:(identities[same[i]],frozenset(identities[v] for v in cross[i])) for i in range(5040)}
 canonical=pairs(same,cross,ids);permutation=torch.randperm(5040,generator=g).tolist();args=[[v[i] for i in permutation] for v in (labels,rx,day,ids)];ss,cc=donor_indices(*args);assert canonical==pairs(ss,cc,args[3])
 mapped=[(v+3)%6 for v in labels];ss,cc=donor_indices(mapped,rx,day,ids);assert canonical==pairs(ss,cc,ids)
 mapping={3:8,8:3,4:6,6:4};ss,cc=donor_indices(labels,[mapping[v] for v in rx],day,ids);assert canonical==pairs(ss,cc,ids)
 negative=0
 def rejects(fn):
  nonlocal negative
  try:fn()
  except ValueError:negative+=1
  else:raise AssertionError('Forbidden input/config accepted')
 for xx,yy,co in [(x[:,:,:255],y,.1),(x,y[:1],.1),(x.long(),y,.1),(x,y.long(),.1),(x,y,-.1),(x,y,.11),(x,y,float('nan')),(x*float('nan'),y,.1),(x,y*float('inf'),.1)]:rejects(lambda:mix(xx,yy,co))
 for which,value in [(0,[6]*5040),(1,[1]*5040),(2,[0]*5040),(3,ids[:-1]),(3,[ids[0]]*5040)]:
  v=[labels[:],rx[:],day[:],ids[:]];v[which]=value;rejects(lambda:donor_indices(*v))
 v=[labels[:],rx[:],day[:],ids[:]];v[0][0]=1;rejects(lambda:donor_indices(*v))
 for key,value in [('mix_coefficient',.2),('U_used',True),('target_access',True),('new_training_epochs',20),('fit_RX',[1,3,4,6]),('total_NN_packets',1)]:
  c=d.config();c[key]=value;rejects(lambda:d.validate(c))
 def metrics():return dict(accuracy=.98,macro_F1=.98,raw_correct_retention=.999,**{'by_'+k:{str(v):dict(accuracy=.98) for v in values} for k,values in [('RX',d.FIT),('TX',range(6)),('day',(1,2,3))]})
 raw=metrics();within=metrics();crossscore=metrics();assert qualify(raw,within,crossscore,[.02,.03],[1e-8])['decision']=='PASS'
 for dimension in ('RX','TX','day'):
  bad=deepcopy(crossscore);bad['by_'+dimension][next(iter(bad['by_'+dimension]))]['accuracy']=.9;assert qualify(raw,within,bad,[.02,.03],[1e-8])['decision']=='FAIL'
 bad=deepcopy(crossscore);bad['raw_correct_retention']=.98;assert qualify(raw,within,bad,[.02,.03],[1e-8])['decision']=='FAIL'
 assert qualify(raw,within,crossscore,[0.,0.],[0.])['decision']=='FAIL'
 return dict(status='PASS',negative_cases=negative,all5040synthetic_metadata_pairing=True,physical_order_TX_RX_permutation_invariant=True,known_phase_gain_frequency_alignment=True,donor_common_phase_gain_invariant=True,non_mutating_rng_preserving=True,common_equivariance_max_packet_relative_RMS=common_error,donor_equivariance_max_packet_relative_RMS=donor_error,relative_RMS_bound=max(info['relative_IQ_change']),relative_power_error=max(info['relative_power_error']),scientific_counterexamples=5,synthetic_IQ_operator_packets=operator_packets,zero_coefficient_packets=zero_coefficient_packets,source_IQ_reads=0,model_forwards=0,optimizer_updates=0,target_access=False)
if __name__=='__main__':print(run())
