"""Frozen source joint160 export and Sinc-only restoration; no fitting."""
import argparse,copy,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .predict import own_model,source_proof
from experiments.cvs_feature_diagnostics.feature_change import measure

@torch.no_grad()
def sinc_restoration(initial,final,source):
 if any(m.training or any(p.requires_grad for p in m.parameters()) for m in (initial,final)):raise ValueError('Source restoration requires frozen eval models')
 state={k:v.clone() for k,v in final.state_dict().items()}
 restored=copy.deepcopy(final)
 restored.id_backbone.encoder.core.id_backbone.sinc.load_state_dict(initial.id_backbone.encoder.core.id_backbone.sinc.state_dict(),strict=True)
 values=[];old=[];logits=[];old_logits=[]
 for begin in range(0,len(source['x']),64):
  x=source['x'][begin:begin+64];a=final.forward_identity_only(x);b=restored.forward_identity_only(x)
  values.append(a['id_feat_joint']);old.append(b['id_feat_joint']);logits.append(a['tx_logits']);old_logits.append(b['tx_logits'])
 z=torch.cat(values);before=torch.cat(old);p=torch.cat(logits);q=torch.cat(old_logits);y=source['y']
 def margin(v):
  wrong=v.clone();wrong[torch.arange(len(y)),y]=-torch.inf
  return v[torch.arange(len(y)),y]-wrong.max(1).values
 delta=(z-before).norm(dim=1);cos=F.cosine_similarity(z,before,dim=1,eps=1e-4);dm=margin(p)-margin(q)
 records=[]
 for dimension,levels in [('overall',[None]),('receiver',[1,3,4,6,8]),('day',[1,2,3])]:
  for level in levels:
   keep=torch.ones(len(y),dtype=torch.bool) if level is None else source['rx' if dimension=='receiver' else 'day']==level
   if not keep.any():continue
   records.append(dict(dimension=dimension,stratum=level,count=int(keep.sum()),final_to_initial_Sinc_joint160_L2_mean=float(delta[keep].mean()),cosine_mean=float(cos[keep].mean()),changed_feature_elements=int((z[keep]!=before[keep]).sum()),default_margin_final_minus_restored_mean=float(dm[keep].mean()),default_accuracy_final=float((p[keep].argmax(1)==y[keep]).float().mean()),default_accuracy_initial_Sinc_restored=float((q[keep].argmax(1)==y[keep]).float().mean())))
 if any(not torch.equal(v,final.state_dict()[k]) for k,v in state.items()):raise ValueError('Frozen source final model changed')
 return dict(status='VERIFIED',records=records,restored='Sinc module only on disposable final model; all downstream final states/default W retained',independent_retrained_ablation=False,source_only=True,query_read=False,model_state_unchanged=True,head_fit=False,optimizer_steps=0,interpretation='Conditional source sensitivity to actual learned filters; not standalone causal DG gain')

def run(seed=None):
 from experiments.cvs_nocfo_foundation import design as nd
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_feature_diagnostics.time_scales import sample_labeled
 from experiments.cvs_equivariant_identity.precision import numerical_context
 torch.set_num_threads(2);L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None
 source=sample_labeled(L);assert len(source['x'])==720
 rows=[];missing=[];start=time.time()
 with numerical_context(d.FULL_FP32_POLICY):
  for row in d.training_rows():
   if seed is not None and row['model_seed']!=seed:continue
   c=d.config(row);p=Path(c['output_root'])
   if not (p/'source_frozen.json').is_file():missing.append(row['row_id']);continue
   source_proof(c);initial=own_model(c,'cpu');final=own_model(c,'cpu')
   initial.load_state_dict(torch.load(p/'initial_smoke.pt',map_location='cpu',weights_only=False)['model'],strict=True)
   rows.append(dict(row_id=row['row_id'],arm=row['arm'],seed=row['model_seed'],source_roles='EXACT_MATCH',epochs=50,feature_boundary=measure(initial,final,source),Sinc_restoration=sinc_restoration(initial,final,source)))
 return dict(status='VERIFIED' if not missing else 'PARTIAL',run_id=d.RUN,models=len(rows),missing=missing,rows=rows,head_fit=False,head_updates=0,query_read=False,source_IQ='fixed L720, first8 physical IDs per TX/RX/day; no U IQ or TX read',seconds=time.time()-start)
def checks():
 from .model import build
 c=d.config(d.training_rows()[1]);torch.set_num_threads(2);torch.manual_seed(c['model_seed'])
 initial=build(c).eval().requires_grad_(False);final=copy.deepcopy(initial)
 with torch.no_grad():final.id_backbone.encoder.core.id_backbone.sinc.low_coordinate.add_(.01)
 source=dict(x=torch.randn(6,2,256),y=torch.arange(6),rx=torch.tensor([1,3,4,6,8,1]),day=torch.tensor([1,2,3,1,2,3]))
 result=sinc_restoration(initial,final,source)
 assert result['records'][0]['final_to_initial_Sinc_joint160_L2_mean']>0
 return dict(status='PASS',synthetic_only=True,conditional_Sinc_sensitivity_detected=True,actual_source_or_query_read=False,head_fit=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--checks',action='store_true');a=p.parse_args();print(json.dumps(checks() if a.checks else run(a.seed)))
