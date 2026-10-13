from . import design as d,model
from experiments.cvs_feature_sequence50 import provenance as shared
def source_proof(c):return shared.source_proof(c,design=d,report_validator=model.validate_report)
def own_model(c,device):return shared.own_model(c,device,design=d,proof=source_proof,builder=model.build)

# Source observations are part of the declared scientific evidence, not a head fit.
_original_source_proof=source_proof
def source_proof(c):
 p=_original_source_proof(c)
 if not c['baseline_reuse']:
  import json
  records=[json.loads(s) for s in (p/'source_mechanism_epochs.jsonl').read_text().splitlines()]
  if len(records)!=50 or [r['epoch'] for r in records]!=list(range(1,51)):raise ValueError('Incomplete mechanism epochs')
  for r in records:
   count=sum(v['total'] for v in r['groups'].values())
   if r['U_truth_read'] or r['gate_changed'] or r['head_fit_updates']!=0 or count!=(56700 if r['epoch']>=34 else 0):raise ValueError('U source observation differs')
  for epoch in (32,33,34,40,50):
   v=d.read(p/f'source_stage_epoch{epoch}.json')
   if v['epoch']!=epoch or v['source']['source_val_count']!=27000:raise ValueError('Source stage snapshot differs')
 return p
