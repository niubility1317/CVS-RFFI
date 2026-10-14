"""Actual composite feature exporter; source-only with no classification fitting."""
import copy,time
from pathlib import Path
import torch
from . import design as d
from .predict import source_proof,own_model
from .features import export
from experiments.cvs_identity_mixstyle50.feature_check import parameter_changes,compare_features
from experiments.cvs_feature_diagnostics.feature_change import measure

def run(seed=None,only_row=None):
 from experiments.cvs_nocfo_foundation import design as nd
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_feature_diagnostics.time_scales import sample_labeled
 from experiments.cvs_equivariant_identity.precision import numerical_context
 torch.set_num_threads(2);L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None;del V;source=sample_labeled(L);rows=[];paired=[];initials={};models={};missing=[];start=time.time()
 with numerical_context(d.FULL_FP32_POLICY):
  for row in d.training_rows():
   if seed is not None and row['model_seed']!=seed or only_row is not None and row['row_id']!=only_row:continue
   c=d.config(row);p=Path(c['output_root'])
   if not (p/'source_frozen.json').exists():missing.append(row['row_id']);continue
   source_proof(c);final=own_model(c,'cpu');initial=copy.deepcopy(final);initial.load_state_dict(torch.load(p/'initial_smoke.pt',map_location='cpu',weights_only=False)['model'],strict=True);initial.eval().requires_grad_(False)
   expected=195041 if c['arm']=='SIG_TIME' else 232033
   change=parameter_changes(initial,final,expected_feature_count=expected)
   added=[v for v in change['records'] if '.sequence_pool.' in v['parameter']]
   if not added or any(v['changed_elements']==0 for v in added):raise ValueError('Added feature parameter did not update numerically')
   rows.append(dict(row_id=row['row_id'],arm=row['arm'],seed=row['model_seed'],feature_boundary=measure(initial,final,source,export_fn=export),actual_initial_to_final_parameters=change))
   models[c['arm']]=final;initials[c['arm']]=initial
  if only_row is None and len(models)==3:
   first=initials['SIG_TIME'].state_dict();native={k:v for k,v in first.items() if '.sequence_pool.' not in k}
   if any(any(not torch.equal(v,m.state_dict()[k]) for k,v in native.items()) for m in initials.values()):raise ValueError('Actual same-seed initial native body/head differs')
   a,b=initials['SIG_TIME_X3'],initials[d.PRIMARY];aa,bb=dict(a.named_parameters()),dict(b.named_parameters())
   if set(aa)!=set(bb) or any(not torch.equal(v,bb[k]) for k,v in aa.items()):raise ValueError('Same-capacity actual initial parameters differ')
   paired.append(dict(seed=d.SEEDS[0],arm=d.PRIMARY,control='SIG_TIME_X3',actual_initial_native_body_and_head_equal=True,actual_initial_parameters_identical=True,source_features=compare_features(models['SIG_TIME_X3'],models[d.PRIMARY],source),actual_final_parameter_difference=parameter_changes(models['SIG_TIME_X3'],models[d.PRIMARY],expected_feature_count=232033)))
 return dict(status='VERIFIED' if not missing else 'PARTIAL',run_id=d.RUN,models=len(rows),missing=missing,rows=rows,paired=paired,source_only=True,head_fit=False,head_updates=0,query_read=False,U_IQ_read=False,U_truth_read=False,V_cache_materialized=True,V_forward=False,source_IQ='fixed L720 first8 perTX/RX/day',seconds=time.time()-start)
