from pathlib import Path
import os,sys,time,json
import torch
from torch.nn import functional as F
from . import design as d
from .learning import fit,predict,probabilities
from experiments.cvs_stage2_rx_actions.model import build
from experiments.cvs_stage2_rx_actions.receiver import ReceiverActions
from experiments.cvs_report_ssl.physical import signal_state
from experiments.cvs_rx_intervention_audit.actions import apply
from experiments.cvs_rx_critic_audit.metrics import margin,summarize
from experiments.cvs_equivariant_identity.precision import numerical_context

def load(c,device):
 from experiments.cvs_stage2_decouple.common import verify_parent
 from experiments.cvs_phase1_stack.recover import verify_contract
 pc=c['parent_build_config'];verify_parent(pc['parent_contract']);prepared=Path(pc['preparation']);done=d.read(prepared/'preparation_complete.json')
 if done['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or done['parent_contract']!=pc['parent_contract'] or done['target_access'] or done['identity_updates']:raise ValueError('Prepared source differs')
 packed=torch.load(prepared/'source_cache.pt',map_location='cpu',weights_only=False);verify_contract(packed['contract'],d.read(d.SOURCE));verify_contract(done['source_contract'],packed['contract'])
 roles={role:{k:packed[key][k] for k in ('x','y','rx','day','ids')} for role,key in [('L','labeled'),('V','validation')]}
 for role,n,contract_role in [('L',6300,'L_s'),('V',27000,'V')]:
  ids=roles[role]['ids']
  if len(ids)!=n or len(set(ids))!=n or sorted(ids)!=sorted(packed['contract']['role_ids'][contract_role]):raise ValueError('Role physical IDs differ')
 if set(roles['L']['ids'])&set(roles['V']['ids']):raise ValueError('L/V overlap')
 model=build(pc,device).eval().requires_grad_(False);bank=ReceiverActions(pc,roles['L']).to(device).eval();bank.critic=None
 return model,bank,roles

@torch.no_grad()
def observe(model,bank,cache,labeled,device):
 arrays={k:[] for k in ('state','delta','CE','margin','clean_correct','action_correct')}
 for start in range(0,len(cache['ids']),256):
  idx=torch.arange(start,min(start+256,len(cache['ids'])));x=cache['x'][idx].to(device);y=cache['y'][idx].to(device);out=model.forward_identity_only(x);logits=out['tx_logits']
  state=torch.cat((signal_state(x),F.normalize(out['id_feat_joint'],dim=1),logits),1)
  if state.shape!=(len(x),206):raise ValueError('Unexpected frozen decision-state dimensions')
  delta=bank.candidates(cache,idx,labeled);base_ce=F.cross_entropy(logits,y,reduction='none');base_m=margin(logits,y);ces=[];ms=[];correct=[]
  for action in range(8):
   changed,_=apply(x,delta[:,action],1.,33,3.);z=model(changed);ces.append(F.cross_entropy(z,y,reduction='none')-base_ce);ms.append(base_m-margin(z,y));correct.append(z.argmax(1)==y)
  values=dict(state=state,delta=delta,CE=torch.stack(ces,1),margin=torch.stack(ms,1),clean_correct=logits.argmax(1)==y,action_correct=torch.stack(correct,1))
  for key,v in values.items():arrays[key].append(v.cpu())
 return dict(**{k:torch.cat(v) for k,v in arrays.items()},ids=list(cache['ids']),rx=cache['rx'],day=cache['day'])

def metrics(pred,data,arm):
 results={}
 for policy,standardized in [('raw',False),('standardized',True)]:
  probs=probabilities(pred,standardized);record={}
  for target,tolerance in [('CE',1e-6),('margin',.01)]:
   x=summarize(pred,data[target],probs,tolerance)
   if not (arm.startswith('CE_') and target=='CE' or arm.startswith('MARGIN_') and target=='margin'):
    for key in ('MSE','zero_prediction_MSE','MSE_relative_to_zero'):x.pop(key)
   record[target]=x
  record['selected_action_accuracy']=float((data['action_correct'].double()*probs).sum(1).mean());record['uniform_action_accuracy']=float(data['action_correct'].double().mean());results[policy]=record
 return results

def run(row_id,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==row_id));d.validate(c);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.perf_counter()
 with numerical_context(d.parent.FULL_FP32_POLICY):
  model,bank,roles=load(c,device);initial={k:v.clone() for k,v in model.state_dict().items()};before={k:v.clone() for k,v in bank.state_dict().items()}
  d.write(out/'resolved_config.json',dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),U_file_opened=False,target_loaders_constructed=False))
  print('RESOLVED',json.dumps(c),flush=True)
  # No V input is used for training or normalization; all fits finish before V observation.
  L=observe(model,bank,roles['L'],True,device);torch.save(L,out/'L_observations.pt');learners={};results={}
  for arm in d.ARMS:
   learner,n,records=fit(L,arm,c['model_seed'],device,c['auxiliary_budget']['steps']);learners[arm]=(learner,n)
   torch.save(dict(model=learner.state_dict(),normalizer=n,arm=arm,config=c),out/(arm+'_auxiliary.pt'))
   (out/(arm+'_training.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in records),encoding='utf-8')
   results[arm]=dict(parameters=sum(p.numel() for p in learner.parameters()),auxiliary_steps=len(records),L=metrics(predict(learner,n,L,arm,device),L,arm));print('FIT',arm,records[-1],flush=True)
  V=observe(model,bank,roles['V'],False,device);torch.save(V,out/'V_observations.pt')
  for arm,(learner,n) in learners.items():
   pred=predict(learner,n,V,arm,device);torch.save(pred,out/(arm+'_V_predictions.pt'));results[arm]['V']=metrics(pred,V,arm)
   if arm.endswith('STATE'):results[arm]['V_shuffled_condition']=metrics(predict(learner,n,V,arm,device,True),V,arm)
   d.write(out/'learning_results.json',dict(complete=False,arms=results,config=c));print('EVAL',arm,results[arm]['V']['standardized']['margin']['pair_concordance'],flush=True)
  if not all(torch.equal(initial[k],v) for k,v in model.state_dict().items()) or not all(torch.equal(before[k],v) for k,v in bank.state_dict().items()):raise ValueError('Frozen identity/bank changed')
  d.write(out/'learning_results.json',dict(complete=True,arms=results,config=c,identity_updates=0,target_access=False))
  d.write(out/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',row_id=row_id,identity_updates=0,auxiliary_updates=12000,target_access=False,L_count=6300,V_count=27000,candidates=8,arms=6,source_model_unchanged=True,bank_unchanged=True,elapsed_seconds=time.perf_counter()-started))
