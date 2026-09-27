import os
from copy import deepcopy
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from cvsrffi.xuc_fusion.ir_solver import IREGSolver
from cvsrffi.xuc_fusion.ir_objective import register_domain_call


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,scale):ctx.scale=scale;return x.view_as(x)
    @staticmethod
    def backward(ctx,g):return -ctx.scale*g,None


class JointToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(3,5),nn.BatchNorm1d(5),nn.Tanh(),nn.Dropout(.15))
        self.tx_head=nn.Linear(5,3)
        self.adv_head=nn.Sequential(nn.Linear(5,7),nn.ReLU(inplace=True),nn.Dropout(.3),nn.Linear(7,2))
    def forward(self,x,grl):
        z=self.encoder(x)
        return self.tx_head(z),self.adv_head(GRL.apply(z,grl))


class External:
    def __init__(self):self.commits=0;self.scale=1.
    def state_dict(self):return dict(commits=self.commits,scale=self.scale)
    def load_state_dict(self,s):self.commits=s['commits'];self.scale=s['scale']


def make_case(implementation='cached',method='IR_EG',kappa=.25,refresh=8):
    device=os.environ.get('IR_ACCEPTANCE_DEVICE','cpu')
    torch.manual_seed(71)
    model=JointToy().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    solver=IREGSolver(model,opt,dict(method=method,implementation=implementation,ir_kappa=kappa,
                                    br_refresh_interval=refresh),max_grad_norm=.7)
    ctx=SimpleNamespace(epoch=21,x=torch.randn(4,3,device=device),u=torch.randn(6,3,device=device),
        y=torch.tensor([0,1,2,1],device=device),d=torch.tensor([0,1,0,1],device=device),
        ud=torch.tensor([0,1,0,1,1,0],device=device),pending=None,field_components=[],forward_calls=0)
    ext=External()
    def closure():
        tx,adv=model(ctx.x,1.);utx,uadv=model(ctx.u,0.)
        task=F.cross_entropy(tx,ctx.y)+.13*utx.square().mean()
        l=F.cross_entropy(adv,ctx.d);u=F.cross_entropy(uadv,ctx.ud)
        divisor=ext.scale if ctx.pending is None else ctx.pending['used']
        if ctx.pending is None:ctx.pending=dict(used=divisor,next=float(task.detach())*.1+.9*ext.scale)
        ctx.field_components.append(['TX','U_task','L_adv','U_adv']);ctx.forward_calls+=2
        if getattr(ctx,'ir_tape',None) is not None:
            register_domain_call(ctx,adv,'L',range(4),ctx.d,.35,1.)
            register_domain_call(ctx,uadv,'U',range(4,10),ctx.ud,.2,0.)
            ctx.ir_assembly=lambda leaves:task/divisor+.35*leaves['L']+.2*leaves['U']
        return task/divisor+.35*l+.2*u
    def commit():ext.scale=ctx.pending['next'];ext.commits+=1
    return SimpleNamespace(model=model,opt=opt,solver=solver,ctx=ctx,closure=closure,external=ext,commit=commit)


def assert_nested(a,b,atol=0.,rtol=0.):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_nested(a[k],b[k],atol,rtol)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_nested(x,y,atol,rtol)
    else:assert a==b


def run(case):
    return case.solver.step(case.closure,ctx=case.ctx,commit=case.commit,external_stateful=(case.external,))
