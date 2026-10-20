"""Source paired GN-output moments and existing-affine initialization only."""
from dataclasses import dataclass
import math
import torch
from torch import nn

def columns(value,positions=16):
    if value.device.type!='cpu' or value.ndim!=3 or value.shape[-1]<positions or not torch.isfinite(value).all():
        raise ValueError('Finite CPU GN output and distinct fixed positions required')
    selected=torch.arange(positions)*(value.shape[-1]-1)//(positions-1)
    return value.detach().index_select(-1,selected).permute(1,0,2).reshape(value.shape[1],-1).double()

class PairedMoment:
    def __init__(self,channels):
        if type(channels)!=int or channels<1:raise ValueError('Positive channel count required')
        self.count=0;self.sums={k:torch.zeros(channels,dtype=torch.float64) for k in ('x','y','xx','xy','yy')}

    @torch.no_grad()
    def update_columns(self,x,y):
        if x.shape!=y.shape or x.ndim!=2 or x.shape[0]!=len(self.sums['x']) or not x.shape[1] or any(v.device.type!='cpu' or v.dtype!=torch.float64 or not torch.isfinite(v).all() for v in (x,y)):
            raise ValueError('Matching finite float64 CPU channel vectors required')
        for key,value in (('x',x),('y',y),('xx',x*x),('xy',x*y),('yy',y*y)):self.sums[key]+=value.sum(1)
        self.count+=x.shape[1]

    def update(self,x,y):
        if x.shape!=y.shape:raise ValueError('Aligned paired GN outputs required')
        self.update_columns(columns(x),columns(y))

    def value(self):
        if not self.count:raise ValueError('Observed paired source outputs required')
        return dict(count=self.count,**{k:v.tolist() for k,v in self.sums.items()})

def affine_fit(value,ridge_fraction,bounds):
    if not math.isfinite(ridge_fraction) or ridge_fraction<=0 or len(bounds)!=2 or not 0<bounds[0]<=1<=bounds[1] or not all(math.isfinite(v) for v in bounds):
        raise ValueError('Positive identity-centered ridge and slope interval containing one required')
    n=value.get('count')
    if type(n)!=int or n<1:raise ValueError('Positive paired moment count required')
    moments={k:torch.tensor(value[k],dtype=torch.float64)/n for k in ('x','y','xx','xy','yy')}
    shape=moments['x'].shape
    if len(shape)!=1 or not shape[0] or any(v.shape!=shape or not torch.isfinite(v).all() for v in moments.values()):raise ValueError('Finite aligned per-channel moments required')
    x,y,xx,xy,yy=(moments[k] for k in ('x','y','xx','xy','yy'))
    var=(xx-x*x).clamp_min(0);cov=xy-x*y;ridge=ridge_fraction*max(float(var.mean()),1e-12)
    slope=((cov+ridge)/(var+ridge)).clamp(*bounds);intercept=y-slope*x
    before=float((xx+yy-2*xy).mean())
    after=float((slope*slope*xx+2*slope*intercept*x+intercept*intercept+yy-2*slope*xy-2*intercept*y).mean())
    scale=max(float(xx.mean()),float(yy.mean()),1e-12)
    if not torch.isfinite(slope).all() or not torch.isfinite(intercept).all() or before < -1e-8*scale or after < -1e-8*scale or after>before+1e-8*scale:
        raise ValueError('GN local affine fit must remain finite and no worse than identity')
    return dict(slope=slope.tolist(),intercept=intercept.tolist(),ridge=ridge,sampled_channel_vectors=n,
        uncorrected_MSE=max(before,0.),fitted_MSE=max(after,0.),slope_clipped_count=int(((slope==bounds[0])|(slope==bounds[1])).sum()))

@torch.no_grad()
def apply_affine(module,item,strength):
    if not isinstance(module,nn.GroupNorm) or not module.affine or type(strength) not in (float,int) or not math.isfinite(strength) or not 0<strength<=1:
        raise ValueError('Existing affine GN and fixed positive correction strength required')
    slope=torch.tensor(item['slope'],dtype=module.weight.dtype,device=module.weight.device)
    intercept=torch.tensor(item['intercept'],dtype=module.bias.dtype,device=module.bias.device)
    if slope.shape!=module.weight.shape or intercept.shape!=module.bias.shape or not torch.isfinite(slope).all() or not torch.isfinite(intercept).all():
        raise ValueError('Matching finite GN coefficients required')
    scale=1+float(strength)*(slope-1)
    module.weight.mul_(scale);module.bias.mul_(scale).add_(float(strength)*intercept)

@dataclass
class NormLink:
    name:str
    source:nn.GroupNorm
    target:nn.GroupNorm
    outputs:torch.Tensor

def norm_links(parent,model,maps):
    links=[]
    for name,source,target in (
        ('coarse_spectral',parent.frozen.coarse.spectral,model.frozen.coarse.spectral),
        ('coarse_temporal',parent.frozen.coarse.temporal,model.frozen.coarse.temporal),
        ('action',parent.frozen.estimator.core,model.frozen.estimator.core),
        *((f'prior{i}',src,dst) for i,(src,dst) in enumerate(zip(parent.prior,model.prior)))):
        selected=maps[name]
        def block(src,dst,keep,prefix):
            for index in (1,4):links.append(NormLink(prefix+f'.net.{index}',src.net[index],dst.net[index],keep))
        for level,(src,dst) in enumerate(zip(source.encoder.blocks,target.encoder.blocks)):
            for index in (0,1):block(src[index],dst[index],selected[level],name+f'.encoder.blocks.{level}.{index}')
        for index,(src,dst) in enumerate(zip(source.decode,target.decode)):
            keep=selected[len(selected)-2-index]
            for number in (0,1):block(src[number],dst[number],keep,name+f'.decode.{index}.{number}')
    if len({id(v.source) for v in links})!=len(links) or len({id(v.target) for v in links})!=len(links):raise ValueError('Existing stage GN modules must remain untied')
    for link in links:
        src,dst,keep=link.source,link.target,link.outputs
        if not all(isinstance(v,nn.GroupNorm) and v.affine and v.num_groups==4 for v in (src,dst)) or src.eps!=dst.eps or keep.shape!=(dst.num_channels,) or keep.dtype!=torch.int64 or keep.device.type!='cpu' or keep.tolist()!=sorted(set(keep.tolist())) or (keep<0).any() or (keep>=src.num_channels).any():
            raise ValueError('Fixed GN4 retained output maps required')
    return links

@torch.no_grad()
def collect(parent,model,maps,views,batch=8):
    if type(batch)!=int or batch<1 or any(v.training or any(p.device.type!='cpu' or p.requires_grad for p in v.parameters()) for v in (parent,model)):
        raise ValueError('Frozen eval CPU parent and initial child required')
    links=norm_links(parent,model,maps);stats={v.name:PairedMoment(len(v.outputs)) for v in links};cache={};seen=set();hooks=[];count=0
    snapshots=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in (parent,model)]
    try:
        for link in links:
            def observe_parent(module,inputs,output,link=link):
                if link.name in cache:raise ValueError('One registered parent GN use per full forward required')
                cache[link.name]=columns(output.index_select(1,link.outputs))
            def observe_child(module,inputs,output,link=link):
                if link.name not in cache or link.name in seen:raise ValueError('Aligned single child GN use required')
                stats[link.name].update_columns(columns(output),cache[link.name]);seen.add(link.name)
            hooks.extend((link.source.register_forward_hook(observe_parent),link.target.register_forward_hook(observe_child)))
        for view in views:
            if view.device.type!='cpu' or view.ndim!=3 or view.shape[1:]!=(2,256) or not torch.isfinite(view).all():raise ValueError('Finite received source IQ required')
            for start in range(0,len(view),batch):
                x=view[start:start+batch];cache.clear();seen.clear();a=parent(x);b=model(x)
                if a.shape!=x.shape or b.shape!=x.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all() or set(stats)!=seen or set(stats)!=set(cache):
                    raise ValueError('Complete finite aligned parent/child source forwards required')
                count+=len(x)
        if not count or any(any(p.grad is not None for p in model.parameters()) or any(not torch.equal(v,old[k]) for k,v in model.state_dict().items()) for model,old in zip((parent,model),snapshots)):
            raise ValueError('Calibration parent or initial child changed')
        from . import design as d
        fits={v.name:dict(outputs=v.outputs.tolist(),**affine_fit(stats[v.name].value(),d.GN_RIDGE_FRACTION,d.GN_SLOPE_BOUNDS)) for v in links}
        if any(v['sampled_channel_vectors']!=count*16 for v in fits.values()):raise ValueError('Full paired source positions required')
        return dict(GN_fits=fits,moments={k:v.value() for k,v in stats.items()},GN_modules=len(links),parent_IQ_inputs=count,student_IQ_inputs=count,
            parent_unchanged=True,initial_student_unchanged=True)
    finally:
        for hook in hooks:hook.remove()
