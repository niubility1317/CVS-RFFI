import json
from pathlib import Path
from . import design as d
def main():
 import torch
 from experiments.cvs_receiver_residual_capacity4.control import process
 from experiments.cvs_frequency_context50.predict import source_proof,own_model
 torch.set_num_threads(2)
 failure=d.read(d.SOURCE_BASE/'failure.json')
 if failure.get('status')!='FAILED' or {r['row_id'] for r in failure.get('rows',[])}!={r['row_id'] for r in d.rows()}:raise ValueError('Original run not fully closed; preserve live rows')
 launch=d.read(d.SOURCE_BASE/'launch.json')
 if any(process(r['pid']) for r in launch['rows']):raise ValueError('Original worker still live')
 artifacts=[]
 for r in d.rows():
  c=d.config(r);source_proof(c)
  # Failure happened before predictions existed. Never duplicate/overwrite any.
  if (d.SOURCE_BASE/r['row_id']/'prediction').exists() or (d.SOURCE_BASE/r['row_id']/'scores').exists():raise ValueError('Original prediction exists; reconcile first')
  log=(d.SOURCE_BASE/'logs'/(r['row_id']+'.log')).read_text()
  if "ModuleNotFoundError: No module named 'baseline_origin_sat_view'" not in log:raise ValueError('Recovery differs from preregistered loading failure')
  model=own_model(c,'cpu')
  with torch.no_grad():out=model.forward_identity_only(torch.zeros(2,2,256))
  if out['tx_logits'].shape!=(2,6) or out['id_feat_joint'].shape!=(2,160) or not torch.isfinite(out['tx_logits']).all():raise ValueError('Own final checkpoint smoke failed')
  artifacts.append(dict(row_id=r['row_id'],source=c['output_root'],epoch=50,updates=11100,strict_loaded=True))
  del model
 result=dict(status='PASS',query_IQ_or_truth_read=False,training_repeated=False,source_processes_absent=True,original_predictions_absent=True,checkpoint_provenance='original scratch50 exact source roles final_only',rows=artifacts)
 p=d.ROOT/'preflight.json'
 if p.exists():raise FileExistsError('Existing preflight')
 d.write(p,result);print(json.dumps(result))
if __name__=='__main__':main()
