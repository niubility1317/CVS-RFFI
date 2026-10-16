"""Frozen-checkpoint gradients on legal source L only; no optimizer or head fit."""
import argparse,json,math,time,random
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from experiments.cvs_rdrop_sinc_units import design as d,model as api
from experiments.cvs_rdrop_sinc_units.provenance import source_proof,validate_checkpoint
from experiments.cvs_sigtime_rdrop_legacy.model import terms
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_phase1_stack.fast_execution import configure
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
SEED=2026092701
PREFIX='id_backbone.encoder.core.id_backbone.sinc.'
W='id_backbone.encoder.core.id_backbone.cls_head.weight'

def moments(a,b):
 a=a.double();b=b.double();na=float(a.norm());nb=float(b.norm());dot=float(a@b)
 return dict(CE_norm=na,KL_norm=nb,dot=dot,cosine=dot/(na*nb) if na and nb else None,KL_to_CE_norm=nb/na if na else None,conflict=bool(dot<0),total_to_CE_dot=float((a+b)@a),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
def verify_state(m,saved):
 if any(not torch.equal(v,m.state_dict()[n]) for n,v in saved.items()):raise ValueError('Diagnostic changed parameters or buffers')
 if any(p.grad is not None for p in m.parameters()):raise ValueError('Diagnostic accumulated parameter grads')
def gather(m,losses,named,scale):
 params=[p for _,p in named];ce,kl=losses
 a=torch.autograd.grad(ce,params,retain_graph=True,allow_unused=False)
 b=torch.autograd.grad(kl,params,allow_unused=False)
 groups={}
 for group in ('Sinc','time_middle','SIG_TIME','remaining_feature','default_W'):
  va=[];vb=[]
  for (name,_),ca,kb in zip(named,a,b):
   key='Sinc' if name.startswith(PREFIX) else 'default_W' if name==W else 'SIG_TIME' if '.sequence_pool.' in name else 'time_middle' if any('.'+t+'.' in name for t in ('time_down','t1','t2')) else 'remaining_feature'
   if key==group:va.append(ca.detach().reshape(-1).double()/(scale if group=='Sinc' else 1.));vb.append(kb.detach().reshape(-1).double()/(scale if group=='Sinc' else 1.))
  if not va:raise ValueError('Missing actual feature gradient group '+group)
  aa=torch.cat(va);bb=torch.cat(vb);groups[group]=moments(aa,bb)
  if group=='Sinc':raw=dict(CE_physical_Hz_gradient=aa.cpu(),KL_physical_Hz_gradient=bb.cpu())
 if len(raw['CE_physical_Hz_gradient'])!=48 or any(not v['finite'] for v in groups.values()):raise ValueError('Invalid actual physical Sinc gradients')
 return groups,raw

def main(here):
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 expected=Path(d.PROJECT)/'runs'/'20261016-phase1-sinc-rdrop-gradient-source-s2026092701-m4-r02'
 if out!=expected or out.exists():raise ValueError('New immutable source-only root required')
 out.mkdir();started=time.time();torch.set_num_threads(2);device=torch.device('cuda:0')
 cases=[];smoke=[]
 with numerical_context(d.FULL_FP32_POLICY):
  for arm in ('A','E'):
   c=d.config(next(r for r in d.rows() if r['arm']==arm and r['model_seed']==SEED));source=source_proof(c)
   ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
   if arm=='E':validate_checkpoint(ck,c)
   else:
    actual=d.read(source/'completion.json')['config']
    if ck.get('run_id')!=actual['run_id'] or ck.get('candidate_id')!=actual['row_id'] or ck.get('epoch')!=50 or ck.get('checkpoint_selection')!='final_only' or ck.get('a1_ema_successful_updates')!=11100:raise ValueError('Original A fixed checkpoint mismatch')
   initial=torch.load(source/'initial_smoke.pt',map_location='cpu',weights_only=False)
   if not initial.get('scratch_only'):raise ValueError('Unknown initial ancestor')
   for stage,state in (('initial',initial['model']),('final50',ck['model'])):
    m=api.build(c,'cpu');m.load_state_dict(state,strict=True);m.eval().to(device)
    with torch.no_grad(),fresh_parameter_caches(m):v=m.forward_identity_only(torch.zeros(2,2,256,device=device))['tx_logits']
    if v.shape!=(2,6) or not torch.isfinite(v).all():raise ValueError('Real-checkpoint no-query smoke failed')
    smoke.append(dict(arm=arm,stage=stage,status='PASS',query_read=False,head_fit_updates=0))
    del m
   cases.append((arm,c,source,initial['model'],ck['model']))
  (out/'real_checkpoint_no_query_smoke.json').write_text(json.dumps(smoke,indent=2)+'\n')
  print('SMOKE PASS actual checkpoint; no source/target IQ or fitted heads',flush=True)
  ec=cases[-1][1];args=configure(d.make_args(ec,'cuda:0'))
  with installed(ec) as native:
   ctx=native._build_ssdg_wisig_data(args,device)
   from cvsrffi.xuc_fusion.native import role_ids_from_native
   if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=d.read(d.SOURCE)['role_ids']:raise ValueError('Source physical roles or query construction mismatch')
   dataset=ctx['train_loader'].dataset
   if len(dataset)!=6300 or len(ctx['unlabeled_loader'].dataset)!=56700 or len(ctx['val_loader'].dataset)!=27000:raise ValueError('Original source roles changed')
   records=[];vectors=[]
   for arm,c,source,initial,final in cases:
    for stage,state,epoch in (('initial',initial,1),('final50',final,50)):
     rowout=out/f'{arm}-{stage}-s{SEED}';rowout.mkdir()
     m=api.build(c,'cpu');m.load_state_dict(state,strict=True);m.to(device).train()
     if m._feature_competition:raise ValueError('Diagnostic may not bypass feature masking')
     named=[(n,p) for n,p in m.named_parameters() if n.startswith('id_backbone.encoder.') and p.requires_grad]
     if dict(named)[W].shape!=(6,160):raise ValueError('Default classifier changed')
     saved={n:v.clone() for n,v in m.state_dict().items()}
     scale=25000. if arm=='E' else 1.
     aug_base=native.build_aug_base_cfg(args);augmentor=native.make_augmentor(aug_base)
     aug_state=native.configure_augmentor_for_epoch(augmentor,aug_base,min(epoch,int(args.label_epochs)),args)
     loader=DataLoader(dataset,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(2026101601),num_workers=0,drop_last=False)
     packets=0
     for batch,(x,y,domain,meta) in enumerate(loader):
      if not bool(((y>=0)&(y<6)).all()):raise ValueError('Only originally labelled L permitted')
      random.seed(2026101602+epoch*1000+batch);np.random.seed(2026101602+epoch*1000+batch);torch.manual_seed(2026101602+epoch*1000+batch);torch.cuda.manual_seed_all(2026101602+epoch*1000+batch)
      x=x.to(device);y=y.to(device);domain=domain.to(device)
      with torch.no_grad():iq=torch.nan_to_num(augmentor(x,labels=y,no_pa=not bool(args.aug_enable_pa_normal)),nan=0.,posinf=0.,neginf=0.)
      with fresh_parameter_caches(m):
       first=m(iq,y_tx=y,grl_lambda=1.,return_aux=True,domain_labels=domain,update_crra_support=False)['tx_logits'];second=m.forward_identity_only(iq,domain_labels=domain)['tx_logits']
       total,ce,kl,_,_=terms(first,second,y,1.)
       groups,raw=gather(m,(ce,kl),named,scale)
      record=dict(arm=arm,stage=stage,batch=batch+1,packets=len(y),CE=float(ce.detach()),KL=float(kl.detach()),groups=groups)
      records.append(record);vectors.append(dict(arm=arm,stage=stage,batch=batch+1,**raw));packets+=len(y)
      with (out/'batch_metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
      with (rowout/'batch_metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
     if packets!=6300 or batch+1!=50:raise ValueError('Complete fixed L pass missing')
     verify_state(m,saved)
     (rowout/'completion.json').write_text(json.dumps(dict(status='ANALYZED',row_id=rowout.name,source_L_packets=packets,batches=batch+1,head_fit_updates=0,optimizer_updates=0,parameter_buffer_changes=0,target_access=False))+'\n')
     print(json.dumps(dict(arm=arm,stage=stage,packets=packets,batches=batch+1,parameter_buffer_changes=0)),flush=True)
     del m,saved
  summary=[]
  for arm in ('A','E'):
   for stage in ('initial','final50'):
    selected=[r for r in records if r['arm']==arm and r['stage']==stage];groups={}
    for group in selected[0]['groups']:
     rows=[r['groups'][group] for r in selected];valid=[r for r in rows if r['cosine'] is not None];ratios=sorted(r['KL_to_CE_norm'] for r in rows if r['KL_to_CE_norm'] is not None)
     groups[group]=dict(batches=len(rows),valid_cosines=len(valid),conflict_count=sum(r['conflict'] for r in rows),conflict_fraction=sum(r['conflict'] for r in rows)/len(rows),cosine_mean=sum(r['cosine'] for r in valid)/len(valid) if valid else None,KL_to_CE_norm_median=(ratios[24]+ratios[25])/2 if len(ratios)==50 else None,total_gradient_opposes_CE_count=sum(r['total_to_CE_dot']<0 for r in rows))
    summary.append(dict(arm=arm,stage=stage,groups=groups,CE_mean=sum(r['CE'] for r in selected)/50,KL_mean=sum(r['KL'] for r in selected)/50))
  torch.save(vectors,out/'Sinc_source_gradient_vectors.pt')
  result=dict(status='ANALYZED',run_id='20261016-phase1-sinc-rdrop-gradient-source-s2026092701-m4-r02',code_commit=(here/'release_commit.txt').read_text().strip(),source_models=['A','E'],model_seed=SEED,checkpoint_stages=['initial','final50'],rows=4,batches=len(records),source_L_packets=25200,unique_L_packets_each_row=6300,identity_packet_forwards=50400,source_only=True,U_IQ_reads=0,U_truth_read=False,V_IQ_reads=0,target_IQ_reads=0,target_truth_reads=0,head_fit_updates=0,optimizer_updates=0,parameter_buffer_changes=0,physical_Hz_gradients=True,augmentation='Original native E1 and min(E50,E33) ordinary augmentation; same actual IQ in paired forwards; matched RNG per arm/stage/batch',source_seen_L=True,limitation='Frozen already-trained source L operation diagnosis; not unseen RX, training causality or performance evidence. No pseudo/U gradient is measured.',summary=summary,elapsed_seconds=time.time()-started)
  (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(out/'completion.json').write_text(json.dumps(dict(status='ANALYZED',rows=4,batches=200,head_fit_updates=0,optimizer_updates=0,target_access=False))+'\n');print(json.dumps(result),flush=True)
