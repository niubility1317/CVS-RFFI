from pathlib import Path
import os,sys,time
import torch
from torch.nn import functional as F
from . import design as d
from .metrics import margin,summarize
from experiments.cvs_stage2_rx_actions.model import build
from experiments.cvs_stage2_rx_actions.receiver import ReceiverActions
from experiments.cvs_rx_intervention_audit.actions import apply
from experiments.cvs_equivariant_identity.precision import numerical_context

def load(c,device):
 from experiments.cvs_stage2_rx_actions.evaluate import source_provenance,checkpoint_provenance
 from experiments.cvs_phase1_stack.recover import verify_contract
 from experiments.cvs_stage2_decouple.common import verify_parent
 pc=c['parent_build_config'];d.parent.validate(pc);source=source_provenance(pc);verify_parent(pc['parent_contract'])
 frozen=d.read(source.parent/'source_frozen.json')
 if frozen['config']!=pc or frozen['target_access'] or frozen['selection']!='final_only':raise ValueError('Source freeze differs')
 checkpoint=torch.load(c['source_checkpoint'],map_location='cpu',weights_only=False);checkpoint_provenance(pc,checkpoint)
 prepared=Path(pc['preparation']);done=d.read(prepared/'preparation_complete.json')
 if done['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or done['parent_contract']!=pc['parent_contract'] or done['target_access'] or done['identity_updates']!=0:raise ValueError('Source cache origin differs')
 packed=torch.load(prepared/'source_cache.pt',map_location='cpu',weights_only=False);verify_contract(packed['contract'],d.read(d.SOURCE));verify_contract(done['source_contract'],packed['contract'])
 keep=('x','y','rx','day','ids');L={k:packed['labeled'][k] for k in keep};V={k:packed['validation'][k] for k in keep}
 for role,cache,n in [('L_s',L,6300),('V',V,27000)]:
  if len(cache['ids'])!=n or len(set(cache['ids']))!=n or sorted(cache['ids'])!=sorted(packed['contract']['role_ids'][role]):raise ValueError('Source physical role differs')
 if set(L['ids'])&set(V['ids']):raise ValueError('L/V overlap')
 model=build(pc,device,c['source_checkpoint']).eval();heads=ReceiverActions(pc,L).to(device).eval()
 auxiliary=torch.load(c['auxiliary_checkpoint'],map_location=device,weights_only=False)
 if auxiliary['teacher'] is not None or auxiliary['teacher_updates']!=0 or auxiliary['selected_for_test']:raise ValueError('Auxiliary source semantics differ')
 heads.load_state_dict(auxiliary['heads'],strict=True)
 return model,heads,L,V

@torch.no_grad()
def evaluate_role(model,heads,cache,labeled,c,device):
 output={k:[] for k in ('predicted_CE','actual_CE','margin_reduction','probabilities','clean_correct','action_correct')}
 for start in range(0,len(cache['ids']),256):
  indices=torch.arange(start,min(start+256,len(cache['ids'])));x=cache['x'][indices].to(device);y=cache['y'][indices].to(device)
  delta=heads.candidates(cache,indices,labeled);clean=model(x);clean_ce=F.cross_entropy(clean,y,reduction='none');clean_margin=margin(clean,y)
  predicted=[];actual=[];margins=[];correct=[]
  for action in range(8):
   changed,_=apply(x,delta[:,action],1.,33,3.);logits=model(changed)
   predicted.append(heads.risks(x,delta[:,action]));actual.append(F.cross_entropy(logits,y,reduction='none')-clean_ce);margins.append(clean_margin-margin(logits,y));correct.append(logits.argmax(1)==y)
  pred=torch.stack(predicted,1);probs=.5/8+.5*(pred/.25).softmax(1)
  values=dict(predicted_CE=pred,actual_CE=torch.stack(actual,1),margin_reduction=torch.stack(margins,1),probabilities=probs,clean_correct=clean.argmax(1)==y,action_correct=torch.stack(correct,1))
  for key,val in values.items():output[key].append(val.cpu())
 output={k:torch.cat(v) for k,v in output.items()};output.update(ids=list(cache['ids']),rx=cache['rx'],day=cache['day'])
 def metrics(mask):
  prob=output['probabilities'][mask];base=output['clean_correct'][mask];act=output['action_correct'][mask]
  margins=summarize(output['predicted_CE'][mask],output['margin_reduction'][mask],prob,.01)
  for key in ('MSE','zero_prediction_MSE','MSE_relative_to_zero'):margins.pop(key)
  return dict(CE=summarize(output['predicted_CE'][mask],output['actual_CE'][mask],prob,1e-6),margin_ranking=margins,clean_accuracy=float(base.double().mean()),uniform_action_accuracy=float(act.double().mean()),policy_action_accuracy=float((act.double()*prob).sum(1).mean()),uniform_break_probability=float((base[:,None]&~act).double().mean()),uniform_fix_probability=float((~base[:,None]&act).double().mean()))
 summary=metrics(torch.ones(len(cache['ids']),dtype=torch.bool));summary['RX']={str(v):metrics(cache['rx']==v) for v in cache['rx'].unique().tolist()};summary['day']={str(v):metrics(cache['day']==v) for v in cache['day'].unique().tolist()}
 summary['margin_calibration_warning']='critic predicts CE units, not margin; no cross-unit MSE is reported; only ranking and selected difficulty are compared'
 return output,summary

def run(row_id,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==row_id));d.validate(c);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.perf_counter()
 with numerical_context(d.parent.FULL_FP32_POLICY):
  model,heads,L,V=load(c,device);before={group:{k:v.detach().clone() for k,v in obj.state_dict().items()} for group,obj in [('model',model),('heads',heads)]}
  d.write(out/'resolved_config.json',dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),target_loaders_constructed=False,U_file_opened=False))
  summaries={}
  for role,cache,labeled in [('L',L,True),('V',V,False)]:
   arrays,summary=evaluate_role(model,heads,cache,labeled,c,device);torch.save(arrays,out/(role+'_actions.pt'));summaries[role]=summary;d.write(out/'ranking_results.json',dict(complete=False,roles=summaries));print(role,summary['CE'],flush=True)
  for group,obj in [('model',model),('heads',heads)]:
   if not all(torch.equal(before[group][k],v) for k,v in obj.state_dict().items()):raise ValueError('Frozen diagnostic mutated model state')
  d.write(out/'ranking_results.json',dict(complete=True,roles=summaries,config=c,identity_updates=0,auxiliary_updates=0,target_access=False,U_used=False))
  d.write(out/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',row_id=row_id,identity_updates=0,auxiliary_updates=0,target_access=False,L_count=6300,V_count=27000,candidates=8,source_model_unchanged=True,auxiliary_unchanged=True,elapsed_seconds=time.perf_counter()-started))
