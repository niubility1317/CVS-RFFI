"""Isolated Dropout replay on real frozen L features; no model training."""
import json,time,traceback
from pathlib import Path
import torch
from torch import nn
from . import design as d
from experiments.cvs_moment_area50.predict import source_proof,own_model
from experiments.cvs_moment_area50.features import shape_descriptor
from experiments.cvs_signature_stats50.descriptors import path_signature
from experiments.cvs_equivariant_identity.precision import numerical_context

def objects(model):
 if any(m.training for m in model.modules()):raise ValueError('Frozen eval model required')
 ref=model.id_backbone.encoder;block=ref.core.id_backbone.t2;pool=ref.sequence_pool
 head=ref.core.id_backbone.cls_head
 if type(head).__name__!='ResidualPhysicalHead' or head.weight.shape!=(6,160) or head.scale!=30:raise ValueError('Default head differs')
 if type(block.drop) is not nn.Dropout or block.drop.p!=.1 or block.pw.out_channels!=96:raise ValueError('Native t2 Dropout differs')
 if pool.arm!='SIG_TIME' or len(pool.views)!=1 or pool.views[0].mode!='SIG_TIME':raise ValueError('Fixed SIG_TIME anchor required')
 model.requires_grad_(False);return ref,block,pool

def select(L,role='L_s'):
 if role!='L_s':raise ValueError('Only L_s permitted')
 if len(L['ids'])!=6300 or set(L['rx'].tolist())!=set(d.RX) or set(L['day'].tolist())!=set(d.DAYS):raise ValueError('Source pool differs')
 from experiments.cvs_feature_diagnostics.time_scales import sample_labeled
 return sample_labeled(L)

@torch.no_grad()
def capture(model,x):
 ref,block,pool=objects(model);pre=[];post=[]
 a=block.pool.register_forward_hook(lambda m,aa,z:pre.append(z.detach().clone()))
 b=block.register_forward_hook(lambda m,aa,z:post.append(z.detach().clone()))
 try:
  z=ref.features(x)
 finally:a.remove();b.remove()
 if len(pre)!=1 or len(post)!=1 or pre[0].shape!=(len(x),96,32) or z.shape!=(len(x),160) or not torch.equal(pre[0],post[0]):raise ValueError('Eval pre/post t2 or actual joint160 differs')
 if not torch.isfinite(z).all() or not torch.isfinite(pre[0]).all():raise ValueError('Nonfinite frozen features')
 return pre[0]

def dropout_replay(h,seed,p=.1):
 if h.ndim!=3 or h.shape[1:]!=(96,32) or p!=.1 or h.device.type!='cpu' or not torch.isfinite(h).all():raise ValueError('Frozen CPU t2 dimensions/probability differ')
 keep=torch.rand(h.shape,generator=torch.Generator().manual_seed(seed))>=p
 return h*keep.to(h.dtype)/(1-p)

@torch.no_grad()
def descriptors(h,pool):
 u=torch.tanh(pool.views[0].local_projection(h)).transpose(1,2)
 b,t,_=u.shape;w=u.reshape(b*4,8,6);sig=path_signature(w,True).reshape(b,4,28)
 ij=torch.triu_indices(7,7,offset=1);feature=ij[1]!=6;time=~feature
 result={'increment6':sig[:,:,:6].flatten(1),'feature_area15':sig[:,:,7:][:,:,feature].flatten(1),'time_area6':sig[:,:,7:][:,:,time].flatten(1),'SIG_TIME28':sig.flatten(1),'COV_SHAPE27':shape_descriptor(w,'COV_SHAPE').reshape(b,108),'AREA_SHAPE27':shape_descriptor(w,'AREA_SHAPE').reshape(b,108),'added_residual160':pool(h)}
 if any(not torch.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite descriptors')
 return result

def statistics(clean,samples,rx,day):
 if len(rx)!=720 or len(day)!=720 or set(clean)!=set(samples):raise ValueError('Fixed source arrays differ')
 records=[]
 for name,base in clean.items():
  noisy=samples[name]
  if base.ndim!=2 or len(base)!=720 or noisy.shape!=(8,*base.shape) or not torch.isfinite(base).all() or not torch.isfinite(noisy).all():raise ValueError('Fixed descriptor arrays differ')
  for dim,values in [('overall',[None]),('receiver',list(d.RX)),('day',list(d.DAYS))]:
   for val in values:
    mask=torch.ones(720,dtype=torch.bool) if val is None else (rx if dim=='receiver' else day)==val
    expected=720 if dim=='overall' else 144 if dim=='receiver' else 240
    if int(mask.sum())!=expected:raise ValueError('Balanced source stratum differs')
    a=base[mask].double();n=noisy[:,mask].double();error=n-a[None];signal=float(a.square().mean().sqrt());noise=float(error.square().mean().sqrt());bias=float(error.mean(0).square().mean().sqrt());spread=float((n-n.mean(0)).square().mean().sqrt())
    records.append(dict(descriptor=name,dimension=dim,stratum=val,count=expected,width=base.shape[1],clean_RMS=signal,noise_RMS=noise,noise_to_clean_RMS=noise/signal if signal>1e-12 else None,mask_mean_bias_RMS=bias,across_mask_RMS=spread,mask_count=8))
 return records

def smoke(model):
 saved={k:v.clone() for k,v in model.state_dict().items()};h=capture(model,torch.zeros(2,2,256));_,_,pool=objects(model);v=descriptors(h,pool);n=descriptors(dropout_replay(h,d.MASK_SEEDS[0]),pool)
 if set(v)!=set(n) or any(not torch.equal(x,model.state_dict()[k]) for k,x in saved.items()):raise ValueError('Real no-query smoke changed state')
 return dict(status='PASS',zero_IQ_packet_forwards=2,query_read=False,head_fit=False,optimizer_steps=0,eval_pre_post_exact=True)

def main():
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_nocfo_foundation import design as nd
 torch.set_num_threads(2);c=d.config(d.rows()[0]);d.validate(c)
 if d.BASE.exists():raise FileExistsError('Immutable root exists; reconcile')
 d.BASE.mkdir(parents=True);out=Path(c['output_root']);out.mkdir();start=time.time()
 try:
  proof=source_proof(c['source_config'])
  if d.read(proof/'resolved_config.json')['commit']!=c['source_execution_commit']:raise ValueError('Fixed source execution differs')
  model=own_model(c['source_config'],'cpu');objects(model);saved={k:v.clone() for k,v in model.state_dict().items()}
  L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None;del V;source=select(L);del L
  with numerical_context(d.FULL_FP32_POLICY):
   h=torch.cat([capture(model,source['x'][j:j+64]) for j in range(0,720,64)]);pool=model.id_backbone.encoder.sequence_pool;clean=descriptors(h,pool);samples={k:[] for k in clean}
   for seed in d.MASK_SEEDS:
    for k,v in descriptors(dropout_replay(h,seed),pool).items():samples[k].append(v)
   samples={k:torch.stack(v) for k,v in samples.items()}
  if any(not torch.equal(v,model.state_dict()[k]) for k,v in saved.items()) or any(p.grad is not None for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('Observer changed model/buffers/mode')
  raw=dict(clean=clean,samples=samples,rx=source['rx'],day=source['day'],config=c);torch.save(raw,out/'observed_descriptors.pt')
  records=statistics(clean,samples,raw['rx'],raw['day']);reloaded=torch.load(out/'observed_descriptors.pt',map_location='cpu',weights_only=False)
  if reloaded['config']!=c or statistics(reloaded['clean'],reloaded['samples'],reloaded['rx'],reloaded['day'])!=records:raise ValueError('Saved descriptor recount differs')
  result=dict(status='VERIFIED',run_id=d.RUN,row_id=d.ROW,model_seed=d.SEED,source_packets=720,source_pool_packets=6300,packet_forwards=720,mask_count=8,mask_samples_are_not_model_seeds=True,records=records,source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,V_cache_materialized=True,V_forward=False,head_fit=False,head_updates=0,optimizer_steps=0,training_epochs=0,model_state_unchanged=True,eval_pre_post_exact=True,saved_descriptor_recount='VERIFIED',source_execution_commit=c['source_execution_commit'],checkpoint=(proof/'final_ssdg.pth').as_posix(),seconds=time.time()-start,scope='Isolated replay of native inverted .1 Dropout after frozen t2.pool, with actual final projection. All other modules eval; not actual training-stream masks/BN/EMA trajectory, causal performance proof or newTX transfer.')
  d.write(out/'results.json',result);d.write(out/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),CUDA_VISIBLE_DEVICES='',CPU_threads=2))
  done=dict(status='ANALYZED',models=1,source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0,source_packets=720,source_descriptor_recount='VERIFIED')
  d.write(out/'completion.json',done);d.write(d.BASE/'completion.json',done);print(json.dumps(done),flush=True)
 except Exception as error:
  d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True));raise
if __name__=='__main__':main()
