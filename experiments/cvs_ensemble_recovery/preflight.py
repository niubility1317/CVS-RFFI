import argparse,json
from pathlib import Path
import torch
from . import design as d
from .source import input_proof
from experiments.cvs_ensemble_final30.predict import own_model

def run():
 torch.set_num_threads(2);records=[]
 for row in d.rows():
  c=d.config(row);p,contract=input_proof(c)
  records.append(dict(row_id=row['row_id'],source=str(p),native_role_comparison=contract['native_role_comparison'],new_updates=0))
 model=own_model(d.config(d.rows()[0])['source_config'],'cpu',repair_reference_copy=True);before={k:t.clone() for k,t in model.state_dict().items()}
 with torch.no_grad():p=model(torch.zeros(4,2,256))
 if p.shape!=(4,6) or not torch.isfinite(p).all() or any(not torch.equal(t,model.state_dict()[k]) for k,t in before.items()):raise ValueError('Actual fixed checkpoint ensemble no-query smoke failed')
 return dict(status='PASS',rows=records,actual_runtime_source_contract=True,new_training_updates=0,real_checkpoint_ensemble_smoke=True,target_access=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status='PASS',new_training_updates=0,target_access=False)))
