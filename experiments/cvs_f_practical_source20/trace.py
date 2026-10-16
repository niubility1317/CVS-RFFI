from contextlib import contextmanager
from types import MethodType
import torch
from torch.nn import functional as F

PARAMETERS=(
 'id_backbone.encoder.core.id_backbone.t1.pw.weight',
 'id_backbone.encoder.core.id_backbone.t2.pw.weight',
 'id_backbone.encoder.core.id_backbone.f_proj.weight',
 'id_backbone.encoder.core.id_backbone.pa_proj.0.weight',
 'id_backbone.encoder.core.id_backbone.pa_gate.net.weight',
 *[f'id_backbone.encoder.sequence_pool.views.{i}.local_projection.weight' for i in range(3)],
 'id_backbone.encoder.response_projection.0.weight',
 'id_backbone.encoder.core.id_backbone.sinc.low_coordinate',
 'id_backbone.encoder.core.id_backbone.sinc.band_coordinate',
 'id_backbone.encoder.core.id_backbone.cls_head.gain',
 'id_backbone.encoder.core.id_backbone.cls_head.weight')

class ActualTrace:
    def __init__(self,model):
        self.model=model;self.ref=model.id_backbone.encoder;self.head=self.ref.core.id_backbone.cls_head
        self.values={};self.representation=None;self.maximum_reconstruction_error=0.
        if len(self.ref.sequence_pool.views)!=3 or self.head.scale!=30. or type(self.head).__name__!='ResidualPhysicalHead':
            raise ValueError('Fixed F three SIG views and original scale30 cosine head required')
    @contextmanager
    def installed(self):
        old=self.head.components;handles=[]
        def components(head,*args,**kwargs):
            result=old(*args,**kwargs)
            if 'core_identity' in self.values:raise ValueError('Multiple identity-head feature paths in one packet forward')
            self.values.update(core_identity=result[0],core_physical=result[1],core_joint=result[2],gated_PA=result[1]*head.gain.tanh())
            return result
        self.head.components=MethodType(components,self.head)
        modules=dict(time_projection=self.ref.core.id_backbone.t_proj,frequency_projection=self.ref.core.id_backbone.f_proj,
            PA_projection=self.ref.core.id_backbone.pa_proj,reference_projection=self.ref.response_projection,
            SIG=self.ref.sequence_pool,**{f'SIG_descriptor_{i}':v for i,v in enumerate(self.ref.sequence_pool.views)})
        core=self.ref.core.id_backbone
        modules['PA_gate_logits']=core.pa_gate.net
        if type(core.freq_gate).__name__=='FreqBandGate1d' and core.freq_gate.alpha>0:modules['frequency_gate_logits']=core.freq_gate.conv
        self.gate_logit_keys=set(modules)-{'time_projection','frequency_projection','PA_projection','reference_projection','SIG',
            'SIG_descriptor_0','SIG_descriptor_1','SIG_descriptor_2'}
        def capture(name):
            def hook(module,args,output):
                if name in self.values:raise ValueError('Feature module unexpectedly called twice: '+name)
                self.values[name]=output
            return hook
        try:
            for name,module in modules.items():handles.append(module.register_forward_hook(capture(name)))
            yield self
        finally:
            self.head.components=old
            for h in handles:h.remove()
    def forward(self,x):
        self.values={};out=self.model.forward_identity_only(x)
        required={'core_identity','core_physical','core_joint','gated_PA','time_projection','frequency_projection','PA_projection',
            'reference_projection','SIG','SIG_descriptor_0','SIG_descriptor_1','SIG_descriptor_2'}
        required|=self.gate_logit_keys
        if set(self.values)!=required:raise ValueError(('Actual branch path incomplete',set(self.values)^required))
        core=self.ref.core.id_backbone
        self.values['PA_envelope_scale']=1+core.pa_gate.alpha*(2*self.values.pop('PA_gate_logits').sigmoid()-1)
        if 'frequency_gate_logits' in self.values:
            self.values['frequency_scale']=1+core.freq_gate.alpha*(2*self.values.pop('frequency_gate_logits').sigmoid()-1)
        self.values['reference']=self.ref.response_gain.tanh()*self.values['reference_projection']
        joint=self.values['core_joint']+self.values['reference']+self.values['SIG'];self.values['joint160']=joint
        z=out['z_id'];logits=out['tx_logits']
        if joint.shape!=z.shape or joint.shape!=(len(x),160) or logits.shape!=(len(x),6):raise ValueError('Actual F joint160/defaulthead boundary differs')
        if torch.allclose(joint,z,atol=2e-5,rtol=2e-5):kind='raw_joint160'
        elif torch.allclose(F.normalize(joint,dim=1),z,atol=2e-5,rtol=2e-5):kind='unit_joint160'
        else:raise ValueError('Captured additive F branches do not construct the actual deployment feature')
        reconstructed=self.head.classify(joint)
        if not torch.allclose(reconstructed,logits,atol=3e-5,rtol=3e-5):raise ValueError('Default logits do not come from captured F joint160')
        self.maximum_reconstruction_error=max(self.maximum_reconstruction_error,float((reconstructed-logits).detach().abs().max()))
        if self.representation is not None and self.representation!=kind:raise ValueError('Feature representation changed by input')
        self.representation=kind
        return out,dict(self.values)
    def gates(self):
        return dict(PA=self.head.gain.detach().tanh().cpu().tolist(),reference=self.ref.response_gain.detach().tanh().cpu().tolist(),
            SIG=(.2*self.ref.sequence_pool.gain_raw.detach().tanh()).cpu().tolist(),
            dynamic_gate_types=dict(PA=type(self.ref.core.id_backbone.pa_gate).__name__,frequency=type(self.ref.core.id_backbone.freq_gate).__name__),
            interpretation='Frozen learned gains are static parameters; per-view weighted activation changes are recorded separately')

def gradient(loss,model):
    named=dict(model.named_parameters());names=list(PARAMETERS)
    frequency='id_backbone.encoder.core.id_backbone.freq_gate.conv.weight'
    if frequency in named:names.append(frequency)
    params=[named[k] for k in names]
    values=torch.autograd.grad(loss,params,retain_graph=False,allow_unused=True)
    return {name:torch.zeros_like(p).flatten().detach() if v is None else v.flatten().detach() for name,p,v in zip(names,params,values)}

def gradient_comparison(clean,other):
    result=[]
    if clean.keys()!=other.keys():raise ValueError('Same actual parameter paths required')
    for name,a in clean.items():
        b=other[name]
        aa=float(a.double().square().sum());bb=float(b.double().square().sum());dot=float((a.double()*b.double()).sum())
        result.append(dict(parameter=name,clean_norm=aa**.5,condition_norm=bb**.5,
            cosine=dot/(aa*bb)**.5 if aa>0 and bb>0 else None))
    return result

def feature_summary(values,clean):
    result={}
    for name,value in values.items():
        if name=='reference_projection':continue
        a=value.detach().flatten(1);b=clean[name].detach().flatten(1)
        result[name]=dict(norm=a.norm(dim=1).cpu().tolist(),cosine_to_clean=F.cosine_similarity(a,b,dim=1,eps=1e-8).cpu().tolist(),
            nonzero=(a.norm(dim=1)>1e-8).cpu().tolist())
        if name.endswith('_scale'):
            result[name].update(mean=a.mean(1).cpu().tolist(),std=a.std(1,unbiased=False).cpu().tolist(),
                minimum=a.min(1).values.cpu().tolist(),maximum=a.max(1).values.cpu().tolist())
    return result
