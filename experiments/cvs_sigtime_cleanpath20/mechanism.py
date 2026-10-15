"""Leave the original t2 main-path dropout; read signature paths before it."""
from torch import nn
from experiments.cvs_feature_sequence50.features import SequenceReference
from experiments.cvs_reference_identity.model import ReferenceCVS

MODES=('ELEMENT','NONE','CLEAN_SIG')

class CleanSignatureReference(SequenceReference):
    def features(self,x):
        local={}
        def capture(module,args):
            if 'h' in local:raise ValueError('One t2 dropout invocation per feature call required')
            local['h']=args[0]
        handle=self.core.id_backbone.t2.drop.register_forward_pre_hook(capture)
        try:base=ReferenceCVS.features(self,x)
        finally:handle.remove()
        if set(local)!={'h'}:raise RuntimeError('Original t2 dropout input was not reached')
        # Preserve both branch gradients through the same t2 backbone.
        return base+self.sequence_pool(local['h'])

def install(reference,mode):
    if mode not in MODES or type(reference) is not SequenceReference:
        raise ValueError('One original SIG_TIME reference and a fixed registered mode required')
    block=reference.core.id_backbone.t2;original=block.drop
    if type(original) is not nn.Dropout or original.p!=.1 or original.inplace:
        raise ValueError('Original non-inplace t2 Dropout(.1) required')
    if mode=='NONE':block.drop=nn.Identity().train(original.training)
    elif mode=='CLEAN_SIG':reference.__class__=CleanSignatureReference
    return dict(mode=mode,main_t2_dropout_p=0. if mode=='NONE' else .1,
        signature_tap='t2.drop input after original pool' if mode=='CLEAN_SIG' else 'original t2 output',
        signature_gradient_detached=False,extra_IQ_forwards=0,extra_rng_draws=0,extra_parameters=0,
        inference_same_weight_equivalence=True,head_structure_changed=False)
