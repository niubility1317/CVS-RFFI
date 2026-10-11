import argparse,torch
from . import design as d
from .utility import sample,direction,metrics,virtual,select_from_L
def check():
 torch.manual_seed(82);model=torch.nn.Linear(3,2);p=list(model.parameters());before=[v.detach().clone() for v in p];x=torch.randn(8,3);y=torch.arange(8)%2;probes=dict(L=(x,y),V=(x+1,y));base={k:metrics(model,*v) for k,v in probes.items()};n=sum(v.numel() for v in p)
 v=direction(torch.randn(n),torch.ones(n));res=virtual(model,p,before,v,.01,probes,base);assert all(torch.equal(a,b) for a,b in zip(p,before)) and set(res)=={'L','V'}
 class Broken(torch.nn.Module):
  def forward(self,x):raise RuntimeError('probe failure')
 try:virtual(Broken(),p,before,v,.01,probes,base)
 except RuntimeError:pass
 else:raise AssertionError('Exception missing')
 assert all(torch.equal(a,b) for a,b in zip(p,before))
 a=[dict(L=dict(CE=2.),V=dict(CE=0.)),dict(L=dict(CE=1.),V=dict(CE=100.))];assert select_from_L(a)==1
 cells=torch.cartesian_prod(torch.arange(5),torch.arange(3),torch.arange(6),torch.arange(4));cache=dict(rx=cells[:,0],day=cells[:,1],y=cells[:,2]);gen=torch.Generator().manual_seed(1);i=sample(cache,[0],[0],3,gen);j=sample(cache,[1,2,3,4],[1,2],2,gen);assert len(i)==18 and len(j)==96 and not torch.isin(i,j).any()
 for row in d.rows():d.validate(d.config(row))
 return dict(status='PASS',exact_restore=True,exception_restore=True,V_cannot_select=True,RX_day_disjoint_sampling=True,rows=3)
def preflight():
 from .run import load,batch
 from experiments.cvs_equivariant_identity.precision import numerical_context
 value=check();c=d.config(d.rows()[0])
 with numerical_context(d.parent.FULL_FP32_POLICY):
  model,heads,roles,p,sc=load(c,'cpu');before={k:v.clone() for k,v in model.state_dict().items()};r=batch(model,heads,roles,p,sc,c,0,'cpu');assert len(r['scales'])==2 and all(torch.equal(before[k],v) for k,v in model.state_dict().items())
 value['actual_E200_8actions_2scales_restored']=True;return value
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--remote-preflight',action='store_true');a=p.parse_args();torch.set_num_threads(2);d.write(a.output,preflight() if a.remote_preflight else check())
