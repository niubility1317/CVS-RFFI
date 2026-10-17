"""Normalization-order controls and focused source/parameter negative cases."""
from copy import deepcopy
import numpy as np
from . import design as d
from .core import F,mean_matrix,packet_prediction,matrix_prediction,evaluate,validate_inputs
from .permissions import validate
def fixture(kind):
 H=np.empty((18,5,20),complex);V=np.empty((5,3,6,20,20),complex);factors=[];meta={k:[] for k in ('rx','day','y')}
 for rxindex,rx in enumerate(d.RX):
  h=np.exp(.12*rxindex*np.cos(np.arange(20))+.3j*rxindex*np.sin(np.arange(20)/3));H[:,rxindex]=h
  for day in range(3):
   for tx in range(6):
    local=np.random.default_rng(202610185102+day*6+tx);z=(local.normal(size=(300,20))+1j*local.normal(size=(300,20)))*np.exp(.3*np.cos(np.arange(20)+tx));z/=np.linalg.norm(z,axis=1,keepdims=True);z*=np.linspace(.2,.95,300)[:,None];z[0]=0
    actual=h if kind=='shared' else (1/h if kind=='reversal' else np.ones(20));w=z*actual;tr=np.sum(abs(z)**2,axis=1);den=np.sum(abs(w)**2,axis=1);w*=np.sqrt(np.divide(tr,den,out=np.zeros_like(tr),where=den>0))[:,None];u=np.fft.ifft(w,axis=1)*np.sqrt(20);V[rxindex,day,tx]=mean_matrix(u);factors.append(u)
    for key,value in [('rx',rx),('day',day+1),('y',tx)]:meta[key].extend([value]*300)
 L=V.copy();old=dict(conditions=dict(phase_products_nonzero=True),folds=[])
 lc=L-L.mean(0,keepdims=True);vc=V-V.mean(0,keepdims=True)
 for day in range(3):
  for tx in range(6):
   k=day*6+tx;days=[a for a in range(3) if a!=day];txs=[a for a in range(6) if a!=tx];actual=vc[:,day,tx];m=np.stack([matrix_prediction(V[0,day,tx],h) for h in H[k]]);m-=m.mean(0);pa=lc[:,days][:,:,txs].mean((1,2));pc=lc[:,days,tx].mean(1)
   old['folds'].append(dict(TX=tx,day=day+1,baseline_SSE=float(np.sum(abs(actual)**2)),additive_SSE=float(np.sum(abs(actual-pa)**2)),TXconditional_SSE=float(np.sum(abs(actual-pc)**2)),response_SSE=float(np.sum(abs(actual-m)**2))))
 return L,V,H,np.concatenate(factors),meta,old
def run():
 count=0
 def reject(fn):
  nonlocal count
  try:fn()
  except (ValueError,KeyError,TypeError):count+=1
  else:raise AssertionError('Invalid fixed-response scope accepted')
 old=dict(status='ARTIFACTS_COMPLETE_SOURCE_ONLY',L=6300,V=27000,source_RX=list(d.RX),source_RX_holdout=None,CP=0,NN=0,updates=0,U_used=False,U_truth_read=False,target_access=False)
 ores=dict(source_roles_exact_match=True,config=dict(source_contract=d.SOURCE,window=[80,160],period=20,IQ_points=256,RX=list(d.RX)))
 response=dict(status='ARTIFACTS_COMPLETE_SOURCE_ONLY',decision='CLOSED_FIXED_SHARED_RESPONSE',source_L=6300,source_V=27000,source_RX=list(d.RX),source_RX_holdout=None,raw_IQ_reads=0,CP=0,NN=0,teacher_NN=0,updates=0,U_used=False,U_truth_read=False,target_access=False,IQ_views_generated=0,automatic_training=False)
 rres=dict(source_handle_exact_match=True,config=dict(derived_input=d.DERIVED,reference_RX=1,phase_anchor_bin=1,epsilon=1e-12,source_RX_holdout=None));validate(old,ores,response,rres)
 for which,key,value in [(0,'L',5580),(0,'source_RX_holdout',1),(0,'target_access',True),(0,'source_RX',[3,4,6,8]),(2,'source_L',5580),(2,'NN',1),(2,'target_access',True),(2,'automatic_training',True),(2,'decision','SOURCE_SHARED_RESPONSE_PREDICTION_ONLY')]:
  args=deepcopy([old,ores,response,rres]);args[which][key]=value;reject(lambda:validate(*args))
 for key,value in [('reference_RX',3),('phase_anchor_bin',2),('epsilon',1e-10),('source_RX_holdout',1),('target_access',True)]:
  cfg=d.config();cfg[key]=value;reject(lambda:d.validate(cfg))
 h=np.ones(20,complex);h[0]=2;z=np.zeros((2,20),complex);z[0,0]=1;z[1,1]=1;u=np.fft.ifft(z,axis=1)*np.sqrt(20);exact=packet_prediction(u,h);proxy=matrix_prediction(mean_matrix(u),h);noncommuting=float(np.linalg.norm(exact-proxy));assert noncommuting>.4
 direct=[]
 for a in u:
  P=F@np.outer(a,a.conj())@F.conj().T;raw=np.diag(h)@P@np.diag(h).conj().T;direct.append(F.conj().T@(raw*np.trace(P).real/np.trace(raw).real)@F)
 direct_error=float(np.max(abs(exact-np.mean(direct,axis=0))));assert direct_error<1e-14
 assert np.max(abs(packet_prediction(u,(2+3j)*h)-exact))<1e-14
 assert np.max(abs(packet_prediction(u,np.ones(20))-mean_matrix(u)))<1e-14
 assert np.array_equal(packet_prediction(np.zeros((2,20),complex),h),np.zeros((20,20)))
 reject(lambda:packet_prediction(u,np.zeros(20)));reject(lambda:packet_prediction(u[:,:19],h));reject(lambda:packet_prediction(np.full((2,20),np.nan),h))
 cases={}
 for name in ('shared','reversal','TX_only'):
  args=fixture(name);result,pred=evaluate(*args);cases[name]=dict(decision=result['decision'],conditions=result['conditions'],packetwise_SSE=result['packetwise_SSE'],mean_proxy_SSE=result['mean_proxy_SSE']);assert all(result['conditions'].values())==(name=='shared'),name
  if name=='shared':shared=args
 L,V,H,u,meta,prior=shared
 reject(lambda:validate_inputs(L[:4],V,H,u,meta));bad=H.copy();bad[0,0,0]=2;reject(lambda:validate_inputs(L,V,bad,u,meta));bad=deepcopy(meta);bad['rx'][0]=9;reject(lambda:validate_inputs(L,V,H,u,bad));bad=deepcopy(prior);bad['folds'][0]['response_SSE']+=.01;reject(lambda:evaluate(L,V,H,u,meta,bad))
 return dict(status='PASS',negative_cases=count,scientific_cases=cases,noncommutativity_Frobenius_difference=noncommuting,direct_matrix_error=direct_error,synthetic_V_factors_generated=3*27000,complete_synthetic_readout_packet_transforms=3*27000,early_rejected_old_control_readout_packet_transforms=1500,unit_packet_transform_vectors=8,direct_outer_product_transforms=2,response_refits=0,real_IQ=0,CP=0,NN=0,updates=0,target_access=False)
if __name__=='__main__':
 import json
 print(json.dumps(run(),allow_nan=False))
