import argparse,torch
from . import design as d
from .learning import Risk,loss,probabilities,normalizer,normalize,fit,predict
def check():
 torch.manual_seed(91);state=torch.randn(32,40);delta=torch.randn(32,8,8);m=Risk(40)
 assert torch.equal(m(state,torch.zeros_like(delta)),torch.zeros(32,8))
 truth=torch.arange(8).float()[None].repeat(32,1)
 assert loss(truth,truth,truth,'RANK')<loss(-truth,truth,truth,'RANK')
 uniform=probabilities(torch.zeros_like(truth),True);assert torch.equal(uniform,torch.full_like(truth,.125))
 assert torch.allclose(probabilities(truth,True),probabilities(truth*4+7,True),atol=1e-6)
 data=dict(state=torch.randn(96,206),delta=torch.randn(96,8,8));data['margin']=data['delta'][:,:,0];data['CE']=.01*data['margin']
 for arm in d.ARMS:
  model,n,logs=fit(data,arm,7,'cpu',steps=6);q=predict(model,n,data,arm,'cpu');assert torch.isfinite(q).all() and q.shape==(96,8) and len(logs)==6
  if arm.endswith('STATE'):assert predict(model,n,data,arm,'cpu',True).shape==(96,8)
 for row in d.rows():
  c=d.config(row);d.validate(c);assert c['source_checkpoint'].startswith('/home/') and '\\' not in c['source_checkpoint']
 return dict(status='PASS',zero_action_anchored=True,ranking_direction=True,scale_free_probability=True,six_arm_forward_backward=True,target_access=False)
def preflight():
 from .run import load,observe
 value=check();evidence=[]
 for row in d.rows():
  c=d.config(row);model,bank,roles=load(c,'cpu');tiny={k:v[:2] for k,v in roles['L'].items()};obs=observe(model,bank,tiny,True,'cpu');assert obs['state'].shape==(2,206)
  idx=torch.arange(2);base=bank.candidates(tiny,idx,False);assert torch.equal(base,bank.candidates(dict(tiny,y=torch.full_like(tiny['y'],999)),idx,False))
  evidence.append(dict(row_id=row['row_id'],real_source_packets=2,all8_actions=True,TX_blind_V_bank=True))
 value['real_E200_checks']=evidence;return value
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--remote-preflight',action='store_true');a=p.parse_args();torch.set_num_threads(2);d.write(a.output,preflight() if a.remote_preflight else check())
