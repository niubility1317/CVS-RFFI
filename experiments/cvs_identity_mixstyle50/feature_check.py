"""Frozen actual f_theta states/joint160; no new reader, fit or query access."""
import argparse,copy,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as default_design
d=default_design
from .predict import own_model,source_proof
from experiments.cvs_feature_diagnostics.feature_change import measure
PREFIX='id_backbone.encoder.core.id_backbone.'
CLASS_W=PREFIX+'cls_head.weight'

@torch.no_grad()
def parameter_changes(a,b,*,expected_feature_count=176065):
 pa={k:v for k,v in a.named_parameters() if k.startswith('id_backbone.')};pb={k:v for k,v in b.named_parameters() if k.startswith('id_backbone.')}
 if set(pa)!=set(pb) or CLASS_W not in pa or pa[CLASS_W].numel()!=960:raise ValueError('Actual identity/default-classifier parameter boundary differs')
 records=[]
 for name,x in pa.items():
  y=pb[name]
  if x.shape!=y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Invalid actual parameter state')
  dx=(y.double()-x.double());norm=float(x.double().norm())
  records.append(dict(parameter=name,elements=x.numel(),changed_elements=int((x!=y).sum()),delta_L2=float(dx.norm()),maximum_abs_change=float(dx.abs().max()),relative_L2=float(dx.norm())/norm if norm else None,scope='class_W' if name==CLASS_W else 'f_theta'))
 theta=[r for r in records if r['scope']=='f_theta'];total=sum(r['elements'] for r in theta)
 if total!=expected_feature_count:raise ValueError('Feature extractor parameter count differs')
 transforms=[r for r in theta if '.cls_head.' in r['parameter']]
 if not transforms:raise ValueError('Feature norms/gain incorrectly removed with class W')
 return dict(status='VERIFIED',feature_parameters=total,class_parameters=960,feature_changed_elements=sum(r['changed_elements'] for r in theta),class_changed_elements=next(r['changed_elements'] for r in records if r['scope']=='class_W'),feature_transform_parameters_preserved=[r['parameter'] for r in transforms],records=records,scope='Only the actual identity subtree; remove class_W alone, keep feature normalization/gain/reference. Per-tensor relative values; mixed Hz/other units are not combined into a causal importance score.',parameter_or_buffer_updates_by_analysis=0)

@torch.no_grad()
def compare_features(a,b,source):
 if any(m.training or any(p.requires_grad for p in m.parameters()) for m in (a,b)):raise ValueError('Only frozen eval feature comparisons allowed')
 saved=[{k:v.clone() for k,v in m.state_dict().items()} for m in (a,b)]
 z=[];q=[]
 for start in range(0,len(source['x']),64):
  x=source['x'][start:start+64];z.append(a.forward_identity_only(x)['id_feat_joint']);q.append(b.forward_identity_only(x)['id_feat_joint'])
 z=torch.cat(z);q=torch.cat(q)
 if z.shape!=q.shape or z.shape[1]!=160 or not torch.isfinite(z).all() or not torch.isfinite(q).all():raise ValueError('Actual joint160 shape/values differ')
 delta=(q-z).norm(dim=1);cos=F.cosine_similarity(z,q,dim=1,eps=1e-4);records=[]
 for dimension,levels in [('overall',[None]),('receiver',[1,3,4,6,8]),('day',[1,2,3])]:
  for value in levels:
   keep=torch.ones(len(z),dtype=torch.bool) if value is None else source['rx' if dimension=='receiver' else 'day']==value
   if keep.any():records.append(dict(dimension=dimension,stratum=value,count=int(keep.sum()),joint160_L2_mean=float(delta[keep].mean()),cosine_mean=float(cos[keep].mean()),changed_feature_elements=int((z[keep]!=q[keep]).sum())))
 if any(not torch.equal(v,m.state_dict()[k]) for m,s in zip((a,b),saved) for k,v in s.items()):raise ValueError('Frozen model state changed')
 return dict(status='VERIFIED',actual_feature='joint160/id_feat_joint',records=records,model_parameters_buffers_unchanged=True,source_only=True,query_read=False,head_fit=False,optimizer_steps=0,interpretation='Same-source frozen feature difference; not newTX transfer or proof that parameter/feature changes cause DG gain')

def run(seed=None,*,design=None,proof_fn=None,build_fn=None):
 d=design or default_design;proof=proof_fn or source_proof;builder=build_fn or own_model
 from experiments.cvs_nocfo_foundation import design as nd
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_feature_diagnostics.time_scales import sample_labeled
 from experiments.cvs_equivariant_identity.precision import numerical_context
 torch.set_num_threads(2);L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None
 del V
 source=sample_labeled(L);assert len(source['x'])==720
 rows=[];paired=[];missing=[];start=time.time()
 with numerical_context(d.FULL_FP32_POLICY):
  for s in d.SEEDS:
   if seed is not None and s!=seed:continue
   models={};initials={}
   for row in d.training_rows():
    if row['model_seed']!=s:continue
    c=d.config(row);p=Path(c['output_root'])
    if not (p/'source_frozen.json').is_file():missing.append(row['row_id']);continue
    proof(c);final=builder(c,'cpu');initial=copy.deepcopy(final)
    initial.load_state_dict(torch.load(p/'initial_smoke.pt',map_location='cpu',weights_only=False)['model'],strict=True);initial.eval().requires_grad_(False)
    rows.append(dict(row_id=row['row_id'],arm=row['arm'],seed=s,source_roles='EXACT_MATCH',epochs=50,feature_boundary=measure(initial,final,source),actual_initial_to_final_parameters=parameter_changes(initial,final)))
    models[row['arm']]=final;initials[row['arm']]=initial
   if set(models)=={'MATCHED_CONTROL','IDENTITY_MIXSTYLE'}:
    aa,bb=initials['MATCHED_CONTROL'].state_dict(),initials['IDENTITY_MIXSTYLE'].state_dict()
    if set(aa)!=set(bb) or any(not torch.equal(v,bb[k]) for k,v in aa.items()):raise ValueError('Actual same-seed initial function/state differs')
    a,b=models['MATCHED_CONTROL'],models['IDENTITY_MIXSTYLE']
    paired.append(dict(seed=s,arm='IDENTITY_MIXSTYLE',control='MATCHED_CONTROL',actual_initial_states_identical=True,source_features=compare_features(a,b,source),actual_final_parameter_difference=parameter_changes(a,b)))
 return dict(status='VERIFIED' if not missing else 'PARTIAL',run_id=d.RUN,models=len(rows),missing=missing,rows=rows,paired=paired,head_fit=False,head_updates=0,query_read=False,U_IQ_read=False,U_truth_read=False,V_cache_materialized=True,V_forward=False,source_IQ='fixed L720 first8 physical IDs/TX/RX/day, actual identity path',seconds=time.time()-start)

def checks():
 from .model import build
 torch.set_num_threads(2);c=d.config(d.training_rows()[0]);torch.manual_seed(c['model_seed']);a=build(c).eval().requires_grad_(False);b=copy.deepcopy(a)
 with torch.no_grad():b.id_backbone.encoder.core.id_backbone.cls_head.weight.add_(.01)
 r=parameter_changes(a,b);assert r['feature_changed_elements']==0 and r['class_changed_elements']==960
 source=dict(x=torch.randn(6,2,256),rx=torch.tensor([1,3,4,6,8,1]),day=torch.tensor([1,2,3,1,2,3]))
 assert compare_features(a,b,source)['records'][0]['changed_feature_elements']==0
 with torch.no_grad():b.id_backbone.encoder.core.id_backbone.cls_head.base_norm.weight.add_(.01)
 r=parameter_changes(a,b);assert r['feature_changed_elements']>0 and any('.cls_head.base_norm.' in n for n in r['feature_transform_parameters_preserved'])
 assert compare_features(a,b,source)['records'][0]['joint160_L2_mean']>0
 try:compare_features(a.train(),b,source)
 except ValueError:pass
 else:raise AssertionError('Train-mode feature observer accepted')
 return dict(status='PASS',synthetic_only=True,class_W_only_changes_excluded=True,feature_head_normalization_kept_and_detected=True,frozen_eval_required=True,source_or_query_IQ_read=False,head_fit=False,updates=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--checks',action='store_true');a=p.parse_args();print(json.dumps(checks() if a.checks else run(a.seed)))
