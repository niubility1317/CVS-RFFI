"""Round-trip closure regression; dataset validator is isolated, no IQ/truth read."""
import argparse,json
from pathlib import Path
from unittest.mock import patch
import torch
from . import design as d
from . import predict
from .budget import final_artifacts
from experiments.cvs_nocfo_foundation.model import build

def run():
 torch.set_num_threads(2);records=[]
 for arm in ('CE50','VAT_FEATURE'):
  row=next(r for r in d.rows() if r['arm']==arm and r['model_seed']==d.SEEDS[0]);c=d.config(row);active=9990 if c['source_U'] else 0
  counts=dict(identity_updates=11100,L_packets=1420800,U_packets=active*128,accepted_U=active*128,teacher_forwards=active,search_forwards=active,search_grad_calls=active,consistency_forwards=active,zero_direction_fallbacks=0)
  source=Path(c['output']);model=build(c['source_loader_config'],'cpu')
  ck,done,frozen=final_artifacts(c,model,counts,56700 if c['source_U'] else 0,1.,177025)
  resolved=dict(config=c,checkpoint_loaded=False,ancestors=[],EMA=False,satellite_generator=False,identity_input_policy=dict(explicit_CFO_identity_input=False),vat=c['vat'],U_iterator=c['source_U'],actual_layers=[dict(level=k) for k in d.LEVELS])
  files={'completion.json':done,'resolved_config.json':resolved,'source_frozen.json':frozen,'source_contract.json':{}}
  with patch.object(d,'read',side_effect=lambda p:files[Path(p).name]),patch('experiments.cvs_phase1_stack.recover.verify_contract',return_value=None),patch('torch.load',return_value=ck):
   assert predict.source_proof(c)==source
   recovered=predict.own_model(c,'cpu')
   with torch.no_grad():z=recovered(torch.zeros(2,2,256))
   assert z.shape==(2,6) and torch.isfinite(z).all()
   for obj,key in ((done,'logged_steps'),(counts,'identity_updates'),(frozen,'updates'),(ck,'updates')):
    before=obj[key];obj[key]=6660
    try:predict.own_model(c,'cpu')
    except ValueError:pass
    else:raise AssertionError('Old30closure accepted')
    finally:obj[key]=before
   before=ck['epoch'];ck['epoch']=30
   try:predict.own_model(c,'cpu')
   except ValueError:pass
   else:raise AssertionError('Oldepoch accepted')
   finally:ck['epoch']=before
  records.append(dict(arm=arm,valid_epoch50_accepted=True,old_epoch30_rejected=True,source_and_checkpoint_roundtrip=True))
 return dict(status='PASS',records=records,query_IQ_or_truth_read=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(v))
