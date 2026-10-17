"""Local synthetic F execution check; no checkpoint/data/training/evaluation job."""
from pathlib import Path

def verify():
 import json
 from collections import Counter
 import torch
 from experiments.cvs_calibration_refresh30 import design as d,model as api
 torch.set_num_threads(2);c=d.config(d.rows()[0]);m=api.build(c,'cpu');calls=[];handles=[]
 normal=(torch.nn.modules.batchnorm._BatchNorm,torch.nn.GroupNorm,torch.nn.LayerNorm,torch.nn.modules.instancenorm._InstanceNorm)
 for name,module in m.named_modules():
  if isinstance(module,normal):handles.append(module.register_forward_hook(lambda mod,inp,out,name=name:calls.append((name,type(mod).__name__))))
 g=torch.Generator().manual_seed(20261017631);x=torch.randn((4,2,256),generator=g);buffers={k:v.detach().clone() for k,v in m.named_buffers()}
 m.train()
 with torch.no_grad():m.forward_identity_only(x)
 train_calls=list(calls);changed=[k for k,v in m.named_buffers() if not torch.equal(v,buffers[k])];calls.clear();m.eval()
 with torch.no_grad():
  a=m.forward_identity_only(x);b=[m.forward_identity_only(x[:2]),m.forward_identity_only(x[2:])]
 logit_error=float((a['tx_logits']-torch.cat([v['tx_logits'] for v in b])).abs().max());feature_error=float((a['id_feat_joint']-torch.cat([v['id_feat_joint'] for v in b])).abs().max())
 for h in handles:h.remove()
 activeBN=[n for n,t in train_calls if t.startswith('BatchNorm') or t.startswith('SyncBatchNorm')]
 if activeBN or changed or not torch.allclose(a['tx_logits'],torch.cat([v['tx_logits'] for v in b]),atol=1e-5,rtol=1e-5) or not torch.allclose(a['id_feat_joint'],torch.cat([v['id_feat_joint'] for v in b]),atol=1e-5,rtol=1e-5):raise ValueError('Batch-axis explanation remains unresolved')
 value=dict(status='NO_ACTIVE_BATCHNORM_STATE_OR_EVAL_BATCH_DEPENDENCE',model='Exact F+CROSS_RX parent architecture/default head/joint160',module_parameter_count=sum(p.numel() for p in m.parameters()),active_normalization_counts=dict(Counter(t for n,t in train_calls)),active_normalizations=[dict(name=n,type=t) for n,t in train_calls],active_BatchNorm=activeBN,changed_persistent_buffers=changed,max_eval_default_logit_batch_split_error=logit_error,max_eval_joint160_batch_split_error=feature_error,real_IQ=0,CP=0,NN_forward_packets=12,optimizer_updates=0,teacher_forwards=0,source_or_target_access=False,scope='Synthetic execution fact;reject only sequential-BN-state explanation,not general batching laws or identity performance')

 return value

if __name__=='__main__':
 import argparse,json
 p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Preserve existing verification output')
 value=verify()
 with a.out.open('x',encoding='utf8') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in value.items() if k!='active_normalizations'},ensure_ascii=False))
