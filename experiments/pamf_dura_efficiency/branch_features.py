"""Read the actual additive features of the frozen original F, without refits."""
import torch

BRANCHES=('identity','PA','response','signature')


def components(reader,iq):
    if reader.training or any(p.requires_grad for p in reader.parameters()):
        raise ValueError('Frozen eval original F required')
    head=reader.core.id_backbone.cls_head
    modules=dict(identity=head.base_norm,PA=head.pa_norm,response=reader.response_projection,signature=reader.sequence_pool)
    observed={};handles=[]
    def hook(name):
        def save(module,args,value):
            if name in observed:raise ValueError('Original F additive branch called more than once: '+name)
            observed[name]=value
        return save
    try:
        for name,module in modules.items():handles.append(module.register_forward_hook(hook(name)))
        actual=reader.features(iq)
    finally:
        for handle in handles:handle.remove()
    if set(observed)!=set(BRANCHES) or any(v.shape!=(len(iq),160) for v in observed.values()):
        raise ValueError('All four original 160D additive features required')
    observed['PA']=head.gain.tanh()[None]*observed['PA']
    observed['response']=reader.response_gain.tanh()[None]*observed['response']
    total=observed['identity']+observed['PA']+observed['response']+observed['signature']
    torch.testing.assert_close(total,actual,atol=2e-6,rtol=2e-6)
    return actual,observed


def replacements(actual,parts,clean,clean_parts):
    if actual.ndim!=2 or actual.shape[1]!=160 or clean.shape!=actual.shape or set(parts)!=set(BRANCHES) or set(clean_parts)!=set(BRANCHES):
        raise ValueError('Matched original F additive source features required')
    return dict(native=actual,**{name:actual-parts[name]+clean_parts[name] for name in BRANCHES},all_clean=clean)
