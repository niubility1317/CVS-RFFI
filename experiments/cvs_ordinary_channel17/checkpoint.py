"""Exact own source-only final20 B00 parameter provenance, no head fitting."""
from pathlib import Path
import torch
from . import design as d

def model(seed):
 if seed not in d.SEEDS:raise ValueError('Unregistered source model')
 source=d.MODELS[seed];src=d.resolve(source['source'])
 init,done,frozen,contract=(d.read(src/name) for name in ('initialization.json','completion.json','source_frozen.json','source_contract.json'))
 if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or init['seed']!=seed:raise ValueError('Unmatched checkpoint ancestry')
 if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['target_access'] or done['checkpoint_sources'] or done['head_fit_updates'] or frozen['epoch']!=20 or frozen['selection']!='final_only' or frozen['target_access']:raise ValueError('Fixedfinal20 source-only checkpoint required')
 if contract['held_RX']!=1 or set(contract['source_receivers'])!=set(d.FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Model source fold differs')
 ck=torch.load(src/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['args']['seed']!=seed or ck['args']['run_id']!=source['run'] or ck['args']['baseline_ckpt'] or ck['args']['teacher_ckpt'] or ck['args']['source_fold_held_rx']!=1 or ck['checkpoint_selection']!='final_only' or ck['a1_ema_successful_updates']!=4440:raise ValueError('Payload provenance differs')
 from experiments.cvs_crossrx_satmix20 import design as md,model as api
 c=md.config(md.rows()[0])
 if any(c[k]!=frozen['config'][k] for k in ('arm','model_seed','epochs','training_mechanism','supcon','fold')) or c['model_seed']!=seed or c['arm']!='B00' or not c['training_mechanism']['CROSS_RX'] or c['training_mechanism']['Ag'] or c['training_mechanism']['As']:raise ValueError('Frozen F+CROSS_RX baseline recipe differs')
 m=api.build(c,'cpu').eval();m.load_state_dict(ck['model'],strict=True);m.requires_grad_(False);native_args=dict(ck['args']);del ck
 ref=m.id_backbone.encoder;head=ref.core.id_backbone.cls_head
 if len(ref.sequence_pool.views)!=3 or ref.sequence_pool.modes!=('SIG_TIME',)*3 or ref.response.contract()['explicit_CFO_identity_input'] is not False or head.weight.shape!=(6,160) or head.scale!=30.:raise ValueError('Default F/head/NoExplicitCFO boundary differs')
 return m,contract,dict(model_seed=seed,source_only=True,scratch_E20_final20=True,checkpoint_sources=[],target_contact=False,role_counts={k:len(v) for k,v in contract['role_ids'].items()},model_api=api.__file__,config=c,head='Existing jointly trained cosine30/6x160'),native_args

def fit_data(trained):
 carrier=d.read(d.resolve(d.SOURCE));prepared=d.read(d.resolve(d.PREPARATION)/'preparation_complete.json')
 if prepared['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared['target_access'] or prepared['identity_updates']!=0:raise ValueError('Source preparation differs')
 packed=torch.load(d.resolve(d.PREPARATION)/'source_cache.pt',map_location='cpu',weights_only=False)
 for actual in (packed['contract'],prepared['source_contract']):
  if any(actual.get(k)!=v for k,v in carrier.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Existing source handles differ')
 L={k:packed['labeled'][k] for k in ('x','y','rx','day','ids')};del packed
 ids=L['ids'];roles=carrier['role_ids']
 if len(ids)!=6300 or len(set(ids))!=6300 or set(ids)!=set(roles['L_s']) or set(ids)&(set(roles['U_s'])|set(roles['V'])):raise ValueError('Only original disjoint L6300 carrier permitted')
 if L['x'].shape!=(6300,2,256) or L['x'].dtype!=torch.float32:raise ValueError('Native float32 IQ256 required')
 mask=L['rx']!=1;v={k:L[k][mask].clone() for k in ('x','y','rx','day')};v['ids']=[s for s,keep in zip(ids,mask.tolist()) if keep];del L
 if len(v['ids'])!=5040 or len(set(v['ids']))!=5040 or set(v['ids'])!=set(trained['role_ids']['L_s']) or not bool(torch.isfinite(v['x']).all()):raise ValueError('Exact legal model-fit L5040 required')
 return v
