import copy,json
from pathlib import Path
from unittest.mock import patch
import torch
from . import design as d,predict
from .model import build
from .source import closing_records,model_details

def run():
 torch.set_num_threads(2);rows=[]
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm and r['model_seed']==d.SEEDS[0]));m=build(c,'cpu');a=d.make_args(c,'cpu');a.num_domains=5;a.input_len=256
  from scripts.train_daot_rc4_baseline import resolved_config
  from experiments.cvs_phase1_stack.source import clean
  done,frozen=closing_records(c,1.,{},d.UPDATES,d.UPDATES)
  resolved=dict(c,**model_details(m));args=clean(resolved_config(a));init=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],target_contact=False,seed=c['model_seed'],source_roles='EXACT_MATCH')
  ck=dict(epoch=50,a1_ema_successful_updates=d.UPDATES,run_id=d.RUN,candidate_id=c['row_id'],checkpoint_selection='final_only',args=vars(a),baseline_args=vars(a).copy(),model=m.state_dict())
  files={'completion.json':done,'resolved_config.json':resolved,'initialization.json':init,'source_frozen.json':frozen,'resolved_native_args.json':args,'source_contract.json':{}}
  with patch.object(d,'read',side_effect=lambda p:files[Path(p).name]),patch('experiments.cvs_phase1_stack.recover.verify_contract',return_value=None),patch('torch.load',return_value=ck):
   recovered=predict.own_model(c,'cpu')
   with torch.no_grad():z=recovered.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
   assert z.shape==(2,6) and torch.isfinite(z).all()
   negatives=0
   for obj,key,value in [(done,'logged_steps',6660),(done,'epoch',30),(frozen,'updates',6660),(ck,'epoch',30),(ck,'a1_ema_successful_updates',6660),(args,'use_ema_teacher',False),(args,'pseudo_temporal_gate',False),(args,'label_epochs',50),(init,'target_contact',True)]:
    old=obj[key];obj[key]=value
    try:predict.own_model(c,'cpu')
    except ValueError:negatives+=1
    else:raise AssertionError('Invalid source closure accepted')
    finally:obj[key]=old
  rows.append(dict(arm=arm,native_checkpoint_roundtrip=True,epoch50_11100accepted=True,invalid_metadata_rejected=negatives))
 from experiments.cvs_phase1_stack.runtime import blind_sample
 sample=(torch.zeros(2,256),4,1,dict(tx=4,tx_i=4,true_tx_i=4,tx_label=4,y_tx=4,rx_i=1,day_i=1,sig_i=3,eq_i=0,base_index=20))
 hidden=blind_sample(sample);assert hidden[1]==-1 and not {'tx','tx_i','true_tx_i','tx_label','y_tx'}&set(hidden[3])
 return dict(status='PASS',rows=rows,truth_blind_U_metadata=True,query_IQ_or_truth_read=False)
if __name__=='__main__':print(json.dumps(run()))
