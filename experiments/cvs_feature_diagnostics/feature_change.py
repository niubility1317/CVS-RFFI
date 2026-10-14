"""Frozen default-model feature boundary and real initial/final change."""
import copy
import torch
from torch.nn import functional as F
from experiments.cvs_transfer_features.model import from_phase1

@torch.no_grad()
def measure(initial,final,source,*,export_fn=None):
 if len(source['x'])!=720 or any(m.training or any(p.requires_grad for p in m.parameters()) for m in (initial,final)):
  raise ValueError('Expected frozen models and fixed labeled source diagnostic sample')
 exporter=export_fn or from_phase1
 original={k:v.clone() for k,v in final.state_dict().items()}
 exported=exporter(final).eval().requires_grad_(False)
 initial_exported=exporter(initial).eval().requires_grad_(False)
 values=[];old_values=[];max_difference=0.
 for start in range(0,720,64):
  x=source['x'][start:start+64]
  normal=final.forward_identity_only(x)['id_feat_joint'];z=exported.forward_features(x);old=initial_exported.forward_features(x)
  if not torch.equal(normal,z):raise ValueError('Real exported feature differs from the default model joint160')
  max_difference=max(max_difference,float((normal-z).abs().max()))
  values.append(z);old_values.append(old)
 z=torch.cat(values);old=torch.cat(old_values)
 # Change W on a disposable model only, and compare the same real source packets.
 altered=copy.deepcopy(final);altered.id_backbone.encoder.core.id_backbone.cls_head.weight.zero_()
 modified=[]
 for start in range(0,720,64):modified.append(altered.forward_identity_only(source['x'][start:start+64])['id_feat_joint'])
 if not torch.equal(z,torch.cat(modified)):raise ValueError('Old class W changes deployment features')
 if any(not torch.equal(v,final.state_dict()[k]) for k,v in original.items()):raise ValueError('Frozen final model changed during diagnostics')
 if any(n.endswith('cls_head.weight') for n,p in exported.named_parameters()):raise ValueError('Old class weights remain in exported extractor')
 delta=(z-old).norm(dim=1);cos=F.cosine_similarity(z,old,dim=1,eps=1e-4);records=[]
 for dimension,levels in [('overall',[None]),('receiver',[1,3,4,6,8]),('day',[1,2,3])]:
  for value in levels:
   keep=torch.ones(720,dtype=torch.bool) if value is None else source['rx' if dimension=='receiver' else 'day']==value
   records.append(dict(dimension=dimension,stratum=value,count=int(keep.sum()),initial_to_final_raw_joint160_L2_mean=float(delta[keep].mean()),initial_to_final_cosine_mean=float(cos[keep].mean()),changed_feature_elements=int((z[keep]!=old[keep]).sum())))
 return dict(status='VERIFIED',source_packets=720,records=records,actual_feature='joint160/id_feat_joint',export_exact=True,maximum_export_difference=max_difference,class_W_zeroing_leaves_features_identical=True,old_class_W_removed_from_export=True,feature_parameters_buffers_unchanged=True,head_fit=False,head_updates=0,feature_updates_by_diagnostic=0,query_read=False,feature_parameters=sum(p.numel() for p in exported.parameters()),limitation='Real feature change and interface independence; not evidence of new-TX transfer or head-free classification benefit')
