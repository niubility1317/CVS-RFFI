import copy,json,sys,torch
from torch.nn import functional as F
from . import design as d,model as api
from .provenance import validate_args,validate_checkpoint
from .evaluate import qualify
def reject(fn):
 try:fn()
 except (ValueError,RuntimeError,TypeError,KeyError):return
 raise AssertionError('Illegal main configuration/provenance accepted')
def main():
 device='cuda:0' if '--cuda' in sys.argv else 'cpu';torch.set_num_threads(2);negative=0
 if device!='cpu':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 assert len(d.rows())==len(d.historical_rows())==12 and d.totals()['new_optimizer_steps']==93240
 for row in d.rows():
  c=d.config(row);args=vars(d.make_args(c,'cpu'));validate_args(args,c)
  assert (args['epochs'],args['label_epochs'],args['pseudo_epochs'])==(row['epochs'],*d.stages(row['epochs'])) and args['seed']==row['model_seed']
  for k,v in [('coupling','INDEPENDENT'),('test_views',['practical_high']),('head_fit_updates',1),('epochs',40)]:
   bad=dict(c);bad[k]=v;reject(lambda:d.validate(bad));negative+=1
  for k,v in [('baseline_ckpt','fold.pt'),('teacher_ckpt','unknown.pt'),('source_fold_held_rx',1),('label_smoothing',0.),('use_aug',False)]:
   bad=dict(args);bad[k]=v;reject(lambda:validate_args(bad,c));negative+=1
  cp=dict(run_id=d.RUN,candidate_id=c['row_id'],epoch=c['epochs'],checkpoint_selection='final_only',a1_ema_successful_updates=c['epochs']*222,ema_model={'present':1},args=args);validate_checkpoint(cp,c)
  for k,v in [('epoch',13),('checkpoint_selection','val_best'),('run_id',d.SCREEN_RUN)]:
   bad=dict(cp);bad[k]=v;reject(lambda:validate_checkpoint(bad,c));negative+=1
 configs=[d.config(r) for r in d.rows() if r['model_seed']==d.SEEDS[0]];models=[api.build(c,device) for c in configs];initial=models[0].state_dict()
 assert all(set(m.state_dict())==set(initial) and all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) for m in models)
 torch.manual_seed(2026101571);x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6;norms={}
 for c,m in zip(configs,models):
  m.train();torch.manual_seed(2026101572)
  with m._phase1_labeled_feature_context():out=m(x,y_tx=y,return_aux=True)
  ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01);addition,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra={},domains=None,optimizer_step=1,identity_loss=ce,labeled_count=128)
  assert metrics['paired_masks/alpha']==c['training_mechanism']['alpha'] and m._legacy_actual_L_input is None and not m._pair_state.cache and m._pair_state.stage=='idle'
  assert all(z['both_drop']==0 and z['calls']==1 for z in m._pair_state.audit['sites'].values())
  names=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight','id_backbone.encoder.sequence_pool.local_projection.weight');named=dict(m.named_parameters());grads=torch.autograd.grad(ce+addition,[named[k] for k in names]+[api.classifier(m).weight]);assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads);norms[c['arm']]=[float(g.norm()) for g in grads]
  teacher=copy.deepcopy(m).eval();assert teacher._pair_state is not m._pair_state and teacher._pair_state.stage=='idle'
 summaries={'E50_ANTI_RDROP':dict(seed_count=3,accuracy_mean=.85)};paired={'E50_ANTI_RDROP minus E50_INDEP_RDROP':dict(seed_count=3,mean_pp=.1,positive_seeds=2)};interactions={'50':dict(seed_count=3,mean_errors=-1)};assert qualify(summaries,paired,interactions)['qualified']
 for key,value in [('accuracy_mean',.83),('mean_pp',0.),('positive_seeds',1),('mean_errors',0)]:
  a,b,c=copy.deepcopy((summaries,paired,interactions));target=a['E50_ANTI_RDROP'] if key=='accuracy_mean' else c['50'] if key=='mean_errors' else b['E50_ANTI_RDROP minus E50_INDEP_RDROP'];target[key]=value;assert not qualify(a,b,c)['qualified']
 print(json.dumps(dict(status='PASS',device=device,protocol_negative_cases=negative,qualification_negative_cases=4,actual_initial_tensors_equal=len(initial),new_models=12,reused_rows=12,all_seed_epoch_stage_args_verified=True,full_source_only=True,default_W_joint=True,actual_body_SIG_defaultW_gradient_norms=norms,synthetic_identity_packets=1024,real_IQ_reads=0,checkpoint_loads=0,optimizer_steps=0,head_fit_updates=0,target_reads=0,unchanged_mask_algorithm_checks='Reuse c803f2b CPU/CUDA58 negatives and complete4 source runs; not repeated')))
if __name__=='__main__':main()
