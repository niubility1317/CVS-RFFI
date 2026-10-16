"""Full-source wiring, fixed budgets, no inherited weights and real feature gradients."""
import copy,json,sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import design as d,model as api,evaluate
from .provenance import validate_args,validate_checkpoint

def reject(fn):
 try:fn()
 except (ValueError,RuntimeError,TypeError,KeyError,FileNotFoundError):return
 raise AssertionError('Illegal main configuration/provenance accepted')

def main():
 device='cuda:0' if '--cuda' in sys.argv else 'cpu';torch.set_num_threads(2);negative=0
 if device!='cpu':
  torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
  torch.backends.cudnn.benchmark=False
 assert len(d.rows())==12 and len(d.historical_rows())==6 and d.totals()['new_optimizer_steps']==93240
 assert all(c['source_falsification'] is None and c['source_selection'] is None for c in map(d.config,d.rows()))
 for row in d.rows():
  c=d.config(row);args=vars(d.make_args(c,'cpu'));validate_args(args,c)
  assert (args['epochs'],args['label_epochs'],args['pseudo_epochs'])==(row['epochs'],*d.stages(row['epochs'])) and args['seed']==row['model_seed']
  for k,v in [('source_selection',{'held_RX':1}),('test_views',['practical_high']),('head_fit_updates',1),('epochs',40)]:
   bad=dict(c);bad[k]=v;reject(lambda:d.validate(bad));negative+=1
  for k,v in [('baseline_ckpt','fold.pt'),('teacher_ckpt','unknown.pt'),('source_fold_held_rx',1),('label_smoothing',0.),('use_aug',False)]:
   bad=dict(args);bad[k]=v;reject(lambda:validate_args(bad,c));negative+=1
  cp=dict(run_id=d.RUN,candidate_id=c['row_id'],epoch=c['epochs'],checkpoint_selection='final_only',a1_ema_successful_updates=c['epochs']*222,ema_model={'present':1},args=args);validate_checkpoint(cp,c)
  for k,v in [('epoch',13),('checkpoint_selection','val_best'),('run_id','source-fold-run')]:
   bad=dict(cp);bad[k]=v;reject(lambda:validate_checkpoint(bad,c));negative+=1
 norms={};initial_count=None
 names=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight','id_backbone.encoder.sequence_pool.local_projection.weight')
 for budget,seed in ((20,d.SEEDS[0]),(50,d.SEEDS[2])):
  configs=[d.config(r) for r in d.rows() if r['epochs']==budget and r['model_seed']==seed]
  models=[api.build(c,device) for c in configs];initial=models[0].state_dict();initial_count=len(initial)
  assert all(set(m.state_dict())==set(initial) and all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) for m in models)
  torch.manual_seed(2026101671);x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6
  logits=[];CE=[];rng=[]
  for c,m in zip(configs,models):
   m.train();m._gain_capture_aux=True;torch.manual_seed(2026101672)
   with m._phase1_labeled_feature_context():out=m(x,y_tx=y,return_aux=True)
   ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
   addition,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=None,domains=None,optimizer_step=1,identity_loss=ce,labeled_count=128)
   assert m._gain_arm==c['mode'] and m._gain_calls==m._legacy_calls==m._feature_l_steps==1 and m._legacy_actual_L_input is None
   named=dict(m.named_parameters());params=[named[k] for k in names]+[api.classifier(m).weight]
   grads=torch.autograd.grad(ce+addition,params,retain_graph=True)
   assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads)
   aux=torch.autograd.grad(m._gain_last_aux,params,allow_unused=True)
   assert aux[-1] is None and all(torch.isfinite(g).all() and g.norm()>0 for g in aux[:-1])
   norms[c['row_id']]=[float(g.norm()) for g in grads]
   logits.append(out['tx_logits'].detach());CE.append(metrics['rdrop_gain/paired_CE']);rng.append((torch.get_rng_state() if device=='cpu' else torch.cuda.get_rng_state()).clone())
   m._gain_last_aux=None
   teacher=copy.deepcopy(m).eval()
   reject(lambda:teacher._source_labeled_feature_hook(model=teacher,labeled_output=out,labels=y,extra=None,domains=None,optimizer_step=2,identity_loss=ce,labeled_count=128));negative+=1
  assert torch.equal(logits[0],logits[1]) and torch.equal(CE[0],CE[1]) and torch.equal(rng[0],rng[1])
 from .source import Observer
 with TemporaryDirectory() as root:
  c=d.config(d.rows()[0]);observer=Observer();observer.initialize(None,c,root)
  audit=dict(api.f.new_audit(),calls=222,packets=222*128,gradient_probe_calls=222)
  dummy=SimpleNamespace(_gain_audit=audit,_feature_l_steps=222,_legacy_calls=222,_gain_calls=222,_legacy_extra_packets=222*128,_legacy_positive_KL_calls=222,_legacy_actual_L_input=None)
  observer.epoch(dummy,None,[{}],root,device)
  assert d.read(Path(root)/'rdrop_gain_progress.json')['actual_steps']==222
  assert d.read(Path(root)/'source_training_mechanism.json')['config']==c
 # New scorer must reject before consulting any truth-side utility.
 with patch.object(evaluate,'preflight',side_effect=ValueError('Incomplete prediction')),patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':''}):
  reject(lambda:evaluate.score(d.rows()[0]['row_id']));negative+=1
 summaries={f'E50_{m}':dict(seed_count=3,accuracy_mean=.85) for m in d.MODES}
 paired={f'E50_{m} minus E50_RDROP':dict(seed_count=3,mean_pp=.1,positive_seeds=2) for m in d.MODES}
 assert all(v['qualified'] for v in evaluate.qualify(summaries,paired)['candidates'].values())
 for key,value in [('accuracy_mean',.83),('mean_pp',0.),('positive_seeds',1)]:
  a,b=copy.deepcopy((summaries,paired));target=a['E50_DYNAMIC'] if key=='accuracy_mean' else b['E50_DYNAMIC minus E50_RDROP'];target[key]=value
  q=evaluate.qualify(a,b);assert not q['candidates']['DYNAMIC']['qualified'] and not q['source_gate_used']
 a=copy.deepcopy(summaries);a['E50_POINT']['seed_count']=2;reject(lambda:evaluate.qualify(a,paired));negative+=1
 reject(lambda:evaluate.aggregate([]));negative+=1
 print(json.dumps(dict(status='PASS',device=device,protocol_negative_cases=negative,qualification_negative_cases=3,actual_initial_tensors_equal=initial_count,new_models=12,reused_rows=6,all_seed_epoch_stage_args_verified=True,full_source_only=True,source_gate_used=False,default_W_joint=True,auxiliary_direct_W_gradient_none=True,actual_body_SIG_defaultW_gradient_norms=norms,actual_CE_logits_RNG_equal=True,synthetic_identity_packets=1024,real_IQ_reads=0,checkpoint_loads=0,optimizer_steps=0,head_fit_updates=0,target_reads=0,unchanged_operator_checks='Reuse complete gain20/center20 source audits and CPU/CUDA geometry checks;not repeated')))
if __name__=='__main__':main()
