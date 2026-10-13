import json,torch
from pathlib import Path
from . import design as d
def qualified():
 p=Path(d.PROJECT)/'runs'/'20261014-phase1-source-neighbor-pair-gate-manysig-m3-r01'
 done=d.read(p/'completion.json');a=d.read(p/'analysis.json')
 if done.get('status')!='ANALYZED' or done.get('models')!=3 or a.get('models')!=3 or not a.get('source_criterion_pass') or len(a['rows'])!=3:raise ValueError('Fixed source diagnostic has not qualified')
 for row in a['rows']:
  if not row['source_criterion_pass'] or row['query_read'] or row['U_truth_read'] or row['U_IQ_read'] or row['head_updates'] or row['optimizer_steps']:raise ValueError('Source qualifier scope differs')
 return dict(run_id=a['run_id'],models=3,status='PASS',source_only=True)
def main():
 from .checks import checks
 from experiments.cvs_feature_multiscale50.predict import source_proof,own_model
 from experiments.cvs_default_head.predict import logits
 torch.set_num_threads(2);integration=checks();q=qualified();c=d.baseline_config(d.rows()[0]);source_proof(c);model=own_model(c,'cpu')
 with torch.no_grad():z=logits(model,torch.zeros(2,2,256))
 if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Real default old checkpoint zero-IQ smoke failed')
 for row in d.rows():d.validate(d.config(row))
 value=dict(status='PASS',integration=integration,source_qualification=q,real_checkpoint_no_query_smoke=True,old_weights_used='smoke and historical comparison only',training_inherits_old_weights=False,query_read=False,head_fit=False)
 if (d.ROOT/'preflight.json').exists():raise FileExistsError('Existing preflight')
 d.write(d.ROOT/'preflight.json',value);print(json.dumps(value))
if __name__=='__main__':main()
