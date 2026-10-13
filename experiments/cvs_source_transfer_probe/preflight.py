import json
from . import design as d
def main():
 import torch
 from .checks import checks
 from experiments.cvs_frequency_context50.predict import source_proof,own_model
 from experiments.cvs_transfer_features.model import from_phase1
 torch.set_num_threads(2);checks();proof=[]
 for r in d.rows():
  c=d.config(r);source_proof(c['source_config']);model=own_model(c['source_config'],'cpu');features=from_phase1(model).eval().requires_grad_(False)
  with torch.no_grad():
   x=torch.zeros(2,2,256);z=features(x);v=model.forward_identity_only(x)
  if z.shape!=(2,160) or not torch.isfinite(z).all() or not torch.equal(z,v['id_feat_joint']) or any(n.endswith('cls_head.weight') for n,_ in features.named_parameters()):raise ValueError('Actual class-free checkpoint smoke differs')
  proof.append(dict(row_id=r['row_id'],strict_loaded=True,old_W_absent=True,features_identical=True))
 result=dict(status='PASS',real_checkpoint_no_query_smoke=True,query_read=False,target_access=False,feature_updates=0,rows=proof)
 p=d.ROOT/'preflight.json'
 if p.exists():raise FileExistsError('Existing preflight')
 d.write(p,result);print(json.dumps(result))
if __name__=='__main__':main()
