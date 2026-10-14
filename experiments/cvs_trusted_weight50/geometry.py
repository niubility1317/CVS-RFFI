"""Nonblocking frozen L720 feature distances; no fitted/readout classifier."""
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as default_design,model as default_model
from experiments.cvs_signature_stats50.predict import own_model
def statistics(v):
 y,rx,day,z=(v[k] for k in ('y','rx','day','z'))
 if z.shape!=(720,160) or set(y.tolist())!=set(range(6)) or set(rx.tolist())!={1,3,4,6,8} or set(day.tolist())!={1,2,3} or not torch.isfinite(z).all():raise ValueError('Fixed L feature shape differs')
 if any(int(((y==t)&(rx==r)&(day==a)).sum())!=8 for t in range(6) for r in (1,3,4,6,8) for a in (1,2,3)):raise ValueError('Balanced fixed source cell count differs')
 unit=F.normalize(z.double(),dim=1);distance=(2-2*(unit@unit.T)).clamp_min(0);tri=torch.ones(720,720,dtype=torch.bool).triu(1);sameTX=y[:,None]==y[None,:];sameRX=rx[:,None]==rx[None,:];sameDay=day[:,None]==day[None,:]
 relations=[('within_TX_same_RX_same_day',sameTX&sameRX&sameDay,2520),('within_TX_cross_RX_same_day',sameTX&~sameRX&sameDay,11520),('between_TX_same_RX_same_day',~sameTX&sameRX&sameDay,14400),('between_TX_cross_RX_same_day',~sameTX&~sameRX&sameDay,57600)]
 records=[]
 for name,keep,expected in relations:
  x=distance[tri&keep]
  if len(x)!=expected:raise ValueError('Source pair relation counts differ')
  records.append(dict(relation=name,pair_count=len(x),normalized_squared_distance_mean=float(x.mean()),distance_median=float(x.median())))
 intra=records[1]['normalized_squared_distance_mean'];inter=records[2]['normalized_squared_distance_mean']
 return dict(records=records,betweenTX_sameRX_over_withinTX_crossRX=inter/intra if intra>1e-12 else None,source_packets=720,no_classifier_fit=True,scope='Source seen RX/day only, same-day unordered pair distances of actual unit-normalized joint160. No predictions or classW/prototype fitting. Pairs overlap; pair count is not independent model seeds or confidence evidence. Geometry is supplementary, no source qualification/parameter/target selection.')
def run(design=None,model_api=None):
 d=design or default_design;api=model_api or default_model
 from experiments.cvs_nocfo_foundation import design as nd
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_feature_diagnostics.time_scales import sample_labeled
 from experiments.cvs_equivariant_identity.precision import numerical_context
 torch.set_num_threads(2);L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None;del V;source=sample_labeled(L);rows=[];packet_forwards=0
 with numerical_context(d.FULL_FP32_POLICY):
  for row in d.rows():
   c=d.config(row);p=d.BASE/row['row_id']/'frozen_source_feature_geometry.pt'
   if p.is_file():raw=torch.load(p,map_location='cpu',weights_only=False)
   else:
    m=own_model(c,'cpu',design=d,build_fn=api.build);before={k:v.clone() for k,v in m.state_dict().items()};parts=[]
    with torch.no_grad():
     for begin in range(0,720,128):parts.append(m.forward_identity_only(source['x'][begin:begin+128])['id_feat_joint'].cpu())
    if any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Frozen model state changed in geometry analysis')
    raw={k:source[k].cpu() for k in ('y','rx','day')};raw.update(z=torch.cat(parts),config=c);torch.save(raw,p);packet_forwards+=720;del m
   if raw['config']!=c:raise ValueError('Fixed saved feature row differs')
   value=statistics(raw);saved=torch.load(p,map_location='cpu',weights_only=False)
   if statistics(saved)!=value:raise ValueError('Saved source feature distance recount differs')
   rows.append(dict(row_id=row['row_id'],arm=row['arm'],seed=row['model_seed'],**value))
 return dict(status='VERIFIED',run_id=d.RUN,rows=rows,source_only=True,U_IQ_read=False,U_truth_read=False,query_read=False,head_fit=False,optimizer_steps=0,extra_source_packet_forwards=packet_forwards,V_cache_materialized=True,V_forward=False,saved_feature_distance_recount='VERIFIED',nonblocking=True)
