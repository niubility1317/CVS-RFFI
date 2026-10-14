"""Fixed full L6300 proxy; no hidden U, target, optimizer or fitted heads."""
import copy,json,time,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .weights import losses
from experiments.cvs_moment_area50.predict import source_proof,own_model
from experiments.cvs_source_gate_audit.gates import physical_metadata,masks
from experiments.cvs_phase1_overlay.model import native_modules
from experiments.cvs_equivariant_identity.precision import numerical_context

def objects(model):
 head=model.id_backbone.encoder.core.id_backbone.cls_head
 if type(head).__name__!='ResidualPhysicalHead' or head.weight.shape!=(6,160) or head.scale!=30 or any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.id_backbone.modules()):raise ValueError('Default head or identity BN differs')
 model.requires_grad_(False);named=dict(model.named_parameters());params=[named[n] for n in d.PARAMETERS]
 if [p.numel() for p in params]!=[3456,6912,9216]:raise ValueError('t1/t2/t3 derivative objects differ')
 for p in params:p.requires_grad_(True)
 return params

def gradient_stats(gl,gu,gw):
 records=[]
 for name,l,u,w in zip(d.PARAMETERS,gl,gu,gw):
  if any(not torch.isfinite(v).all() for v in (l,u,w)):raise ValueError('Nonfinite feature derivative')
  l,u,w=(v.detach().double().flatten() for v in (l,u,w));nl,nu,nw=(float(v.norm()) for v in (l,u,w))
  records.append(dict(parameter=name,elements=len(l),L_gradient_norm=nl,lambdaU_uniform_gradient_norm=nu,lambdaU_weighted_gradient_norm=nw,uniform_U_to_L=nu/nl if nl>1e-12 else None,weighted_U_to_L=nw/nl if nl>1e-12 else None,weighted_uniform_gradient_difference_norm=float((w-u).norm()),weighted_uniform_cosine=float(torch.dot(w,u)/(nu*nw)) if nu>1e-12 and nw>1e-12 else None))
 return records

def statistics(raw):
 y,rx,day,a,correct=(raw[k] for k in ('y','rx','day','accept','teacher_correct'));effective=raw['effective_weight'].double();records=[]
 if len(y)!=6300 or len(a)!=6300 or int(a.sum())==0 or not torch.isfinite(effective).all():raise ValueError('Full L proxy empty or invalid')
 for dim,values in [('overall',[None]),('receiver',d.RX),('day',d.DAYS),('source_class',range(6))]:
  for value in values:
   keep=torch.ones(6300,dtype=torch.bool) if value is None else (rx if dim=='receiver' else day if dim=='day' else y)==value
   selected=keep&a;n=int(keep.sum());accepted=int(selected.sum());hits=int((selected&correct).sum());mass=float(effective[selected].sum())
   records.append(dict(dimension=dim,stratum=value,count=n,accepted=accepted,accepted_correct=hits,coverage=accepted/n,uniform_precision=hits/accepted if accepted else None,weighted_precision=float((effective[selected]*correct[selected]).sum())/mass if mass else None,effective_weight_mass=mass,hardness_mean=float(raw['hardness'][selected].double().mean()) if accepted else None,raw_weight_min=float(raw['raw_weight'][selected].min()) if accepted else None,raw_weight_max=float(raw['raw_weight'][selected].max()) if accepted else None))
 return records

def smoke(model):
 params=objects(model);saved={k:v.clone() for k,v in model.state_dict().items()};model.eval();x=torch.zeros(2,2,256)
 z=model.forward_identity_only(x);pseudo=torch.tensor([0,1]);a=torch.tensor([True,False]);v=losses(z['tx_logits'],pseudo,a)
 gs=torch.autograd.grad(F.cross_entropy(z['tx_logits'],pseudo,label_smoothing=.01),params,retain_graph=True);gu=torch.autograd.grad(v['weighted'],params)
 if any(not torch.isfinite(g).all() for g in (*gs,*gu)) or any(not torch.equal(v,model.state_dict()[k]) for k,v in saved.items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('No-query gradient smoke differs')
 model.requires_grad_(False);return dict(status='PASS',zero_IQ_packet_forwards=2,query_read=False,head_fit=False,optimizer_steps=0,default_head_unchanged=True,feature_gradient_objects=list(d.PARAMETERS))

def main():
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_nocfo_foundation import design as nd
 from experiments.cvs_moment_area50 import design as parent
 torch.set_num_threads(2);c=d.config(d.rows()[0]);d.validate(c)
 if d.BASE.exists():raise FileExistsError('Existing immutable root; reconcile')
 d.BASE.mkdir(parents=True);out=Path(c['output_root']);out.mkdir();start=time.time()
 try:
  proof=source_proof(c['source_config'])
  if d.read(proof/'resolved_config.json')['commit']!=c['source_execution_commit']:raise ValueError('Source execution differs')
  student=own_model(c['source_config'],'cpu');teacher=copy.deepcopy(student);ck=torch.load(proof/'final_ssdg.pth',map_location='cpu',weights_only=False)
  if ck.get('ema_model') is None or ck['epoch']!=50 or ck['a1_ema_successful_updates']!=11100:raise ValueError('EMA source not verified')
  teacher.load_state_dict(ck['ema_model'],strict=True);teacher.eval().requires_grad_(False);del ck
  params=objects(student);saved={k:v.clone() for k,v in student.state_dict().items()};saved_t={k:v.clone() for k,v in teacher.state_dict().items()}
  args=parent.make_args(c['source_config'],'cpu')
  if args.pseudo_domain_gate or not args.pseudo_temporal_gate or args.pseudo_temporal_mode!='batch_neighbor' or args.pseudo_temporal_window!=2 or args.strong_noise_std!=.015 or args.lambda_u!=.16 or args.label_smoothing!=.01:raise ValueError('Old gate/loss recipe differs')
  L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None;del V;meta=physical_metadata(L,'L_s')
  order=torch.randperm(6300,generator=torch.Generator().manual_seed(c['shuffle_seed']));torch.manual_seed(c['dropout_seed']);student.train();parts={k:[] for k in ('y','rx','day','accept','teacher_correct','hardness','raw_weight','effective_weight')};gradients=[];batch_records=[]
  with numerical_context(d.FULL_FP32_POLICY):
   for begin in range(0,6300,256):
    ix=order[begin:begin+256];x=L['x'][ix];metadata={k:v[ix] for k,v in meta.items()};domains=(torch.searchsorted(torch.tensor(d.RX),L['rx'][ix])*3+L['day'][ix]-1).long()
    with torch.no_grad():weak=teacher.forward_identity_only(x)['tx_logits']
    clean=student.forward_identity_only(x)['tx_logits'];strong=student.forward_identity_only(native_modules()._strong_augment(x,args.strong_noise_std))['tx_logits']
    pseudo,accepted,_=masks(native_modules(),weak,strong.detach(),metadata,domains,args);a=accepted['FULL_OLD'];before=a.clone();v=losses(strong,pseudo,a)
    if not torch.equal(before,a) or not torch.equal(v['accept'],a) or v['raw_weight'].requires_grad or (v['raw_weight']<.75).any() or (v['raw_weight']>1.25).any():raise ValueError('Gate/weight contract differs')
    gl=torch.autograd.grad(F.cross_entropy(clean,L['y'][ix],label_smoothing=.01),params);gu=torch.autograd.grad(.16*v['uniform'],params,retain_graph=True);gw=torch.autograd.grad(.16*v['weighted'],params)
    gradient=gradient_stats(gl,gu,gw);gradients.append(tuple(torch.cat([g.detach().flatten() for g in gs]) for gs in (gl,gu,gw)));batch_records.append(dict(batch_index=begin//256,count=len(ix),accepted=int(a.sum()),L_CE=float(F.cross_entropy(clean.detach(),L['y'][ix],label_smoothing=.01)),uniform_U_CE=float(v['uniform'].detach()),weighted_U_CE=float(v['weighted'].detach()),gradient_records=gradient))
    for key,value in dict(y=L['y'][ix],rx=L['rx'][ix],day=L['day'][ix],accept=a,teacher_correct=pseudo==L['y'][ix],hardness=v['hardness'],raw_weight=v['raw_weight'],effective_weight=v['effective_weight']).items():parts[key].append(value.detach().cpu())
  raw={k:torch.cat(v) for k,v in parts.items()};raw.update(config=c,gradient_vectors=gradients,batch_records=batch_records)
  student.eval().requires_grad_(False)
  if any(not torch.equal(v,student.state_dict()[k]) for k,v in saved.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in saved_t.items()) or any(p.grad is not None for m in (student,teacher) for p in m.parameters()):raise ValueError('Frozen observer mutated models or accumulated gradients')
  torch.save(raw,out/'source_weight.pt');records=statistics(raw);reloaded=torch.load(out/'source_weight.pt',map_location='cpu',weights_only=False)
  if reloaded['config']!=c or statistics(reloaded)!=records:raise ValueError('Saved source weight recount differs')
  for batch,gs in zip(batch_records,reloaded['gradient_vectors']):
   splits=[torch.split(g,[3456,6912,9216]) for g in gs]
   if gradient_stats(*splits)!=batch['gradient_records']:raise ValueError('Saved gradient recount differs')
  result=dict(status='VERIFIED',run_id=d.RUN,row_id=d.ROW,model_seed=d.SEED,source_packets=6300,source_batches=25,packet_forwards=18900,source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,V_cache_materialized=True,V_forward=False,training_epochs=0,optimizer_steps=0,head_fit=False,head_updates=0,student_EMA_state_unchanged=True,old_gate_unchanged=True,weight_stopgrad=True,weight_bounds=[.75,1.25],hardness_scale=.1,records=records,batch_records=batch_records,parameters=list(d.PARAMETERS),saved_weight_gradient_recount='VERIFIED',source_execution_commit=c['source_execution_commit'],seconds=time.time()-start,scope='Fixed frozen L6300 proxy with source student Dropout replay/strongnoise. Source labels used for L CE and post-mask accuracy only. Not hiddenU precision/coverage or actual33/34/50 training gradient history, effective weighting or target/newTX transfer proof.')
  d.write(out/'results.json',result);d.write(out/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),CUDA_VISIBLE_DEVICES='',CPU_threads=2))
  done=dict(status='ANALYZED',models=1,source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0,source_packets=6300,source_weight_gradient_recount='VERIFIED');d.write(out/'completion.json',done);d.write(d.BASE/'completion.json',done);print(json.dumps(done),flush=True)
 except Exception as e:d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),outputs_preserved=True));raise
if __name__=='__main__':main()
