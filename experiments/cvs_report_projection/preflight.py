import argparse,json
from pathlib import Path
import torch
from . import design as d
from .checks import run as checks
from experiments.cvs_scratch30_baseline.predict import own_model,source_proof
from experiments.cvs_report_ssl.evaluate import physical_ids
def run():
 torch.set_num_threads(2);result=checks();proofs=[]
 for seed in d.SEEDS:
  c=d.parent(seed);p=source_proof(c);m=own_model(c,'cpu');before={k:v.clone() for k,v in m.state_dict().items()}
  with torch.no_grad():z=m(torch.zeros(2,2,256))
  if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('RealCE30 matching checkpoint noquery smoke')
  proofs.append(dict(seed=seed,checkpoint=str(p/'identity.pth'),source_roles='EXACT_MATCH',final_epoch=30,scratch_ancestors=[],target_training=False))
 ids=physical_ids();return dict(status='PASS',checks=result,real_checkpoint_noquery=proofs,clean_handle='VALIDATED_ONCE',query_index_count=len(ids),query_IQ_or_truth_read=False,sourceVperformance_gate=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status='PASS',clean_only=True,query_IQ_or_truth_read=False)))
