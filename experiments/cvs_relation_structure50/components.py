import torch

def capture(module,inputs,output):
 # A module-local attribute remains independent under EMA deepcopy.
 module._relation_feature=output

def install(model):
 projection=model.id_backbone.encoder.response_projection
 if hasattr(projection,'_relation_capture_installed'):raise ValueError('Duplicate capture')
 projection._relation_feature=None;projection._relation_capture_installed=True
 projection.register_forward_hook(capture)
 return model

def parts(model,out):
 e=model.id_backbone.encoder;h=e.core.id_backbone.cls_head
 base=out['id_feat_cls'];physical=out['id_feat_pa'];reference=e.response_projection._relation_feature
 if reference is None or reference.shape!=base.shape:raise ValueError('Reference feature not captured by same forward')
 pa=h.gain.detach().tanh()*physical;ref=e.response_gain.detach().tanh()*reference
 joint=base+pa+ref
 # Auxiliary features have exact deployment values, but stop gain gradients.
 if not torch.allclose(joint.detach(),out['id_feat_joint'].detach(),atol=1e-6,rtol=1e-6):raise ValueError('Actual additive CVS reconstruction differs')
 return dict(joint=joint,PA=pa,ref=ref)
