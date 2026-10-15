import copy,json,tempfile,math
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d,model as api
from experiments.cvs_sinc_units50.observer import Observer,summarize_snapshots
from experiments.cvs_sinc_units50.optimizer import NAMES
from experiments.cvs_phase1_overlay.model import native_modules
from experiments.cvs_temporal_rsc50.checks import invoke

def checks():
 torch.set_num_threads(2);rows=d.training_rows();models=[api.build(d.config(row),'cpu') for row in rows]
 for seed in d.SEEDS:
  paired=[m for r,m in zip(rows,models) if r['model_seed']==seed];initial=paired[0].state_dict();assert all(all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) for m in paired)
 args=[d.make_args(d.config(r),'cpu') for r in rows]
 def same(a,b):return a==b or isinstance(a,float) and isinstance(b,float) and math.isnan(a) and math.isnan(b)
 diff=[k for k in vars(args[0]) if not same(vars(args[0])[k],vars(args[1])[k])]
 assert set(diff).issubset({'candidate_id','output_dir'})
 negatives=0;native_changes=[]
 for row in rows:
  c=d.config(row);first=d.first.config(next(r for r in d.first.rows() if r['arm']==row['arm']));assert all(c[k]==first[k] for k in d.MECHANISM_KEYS)
  a,b=d.make_args(c,'cpu'),d.first.make_args(first,'cpu');changed={k for k in vars(a).keys()|vars(b).keys() if not same(vars(a).get(k),vars(b).get(k))};assert not changed-{'seed','run_id','candidate_id','output_dir'};native_changes.append(sorted(changed))
  bad=copy.deepcopy(c);bad['sinc_optimizer_units']['sample_rate_Hz']+=1
  try:d.validate(bad)
  except ValueError:negatives+=1
  else:raise AssertionError('Modified Sinc unit recipe accepted')
 gen=torch.Generator().manual_seed(9560);x=torch.randn(12,2,256,generator=gen);y=torch.arange(12)%6;rx=torch.tensor([1,3,4,6,8,1]*2);n=native_modules();actual=[];saved_class=torch.optim.AdamW
 for row,m,a in zip(rows,models,args):
  c=d.config(row);m.train()
  assert a.use_unlabeled and a.use_ema_teacher and a.pseudo_temporal_mode=='batch_neighbor' and a.pseudo_temporal_gate and not a.freeze_backbone and a.label_epochs==33 and a.pseudo_epochs==17
  with tempfile.TemporaryDirectory(prefix='sinc-native-check-') as tmp:
   obs=Observer(c,design=d);obs.initialize(m,c,Path(tmp))
   with obs.native_scope(n):
    opt=torch.optim.AdamW(n._optimizer_parameters(m,None),lr=float(a.lr),weight_decay=float(a.weight_decay))
    for batch in range(1,17):
     opt.zero_grad(set_to_none=True);out=invoke(m,a,x,y,rx);loss=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01);zero=loss*0
     assert obs.observe(model=m,args=a,epoch=1,batch_idx=batch,loss_l=loss,loss_u=zero,loss_ent=zero,active=False,logits=None,pseudo=None,mask=None) is zero
     loss.backward();torch.nn.utils.clip_grad_norm_(n._optimizer_parameters(m,None),5.);opt.step()
   assert torch.optim.AdamW is saved_class and n._trusted_pseudo_loss_hook is None
   assert opt._sinc_unit_steps==16 and obs.steps==16 and obs.U_packets==0 and len(obs.sinc_raw)==1
   r=obs.sinc_raw[0];assert r['step']==16 and all(v['optimizer_step']==16 for v in r['parameters'].values())
   torch.save(obs.sinc_raw,Path(tmp)/'snapshot.pt');same=torch.load(Path(tmp)/'snapshot.pt',map_location='cpu',weights_only=False)
   assert summarize_snapshots(same)==summarize_snapshots(obs.sinc_raw)
   assert all(p.requires_grad for p in (m.id_backbone.encoder.core.id_backbone.sinc.low_hz_,m.id_backbone.encoder.core.id_backbone.sinc.band_hz_,m.id_backbone.encoder.core.id_backbone.cls_head.weight))
   actual.append(dict(row_id=row['row_id'],mode=c['sinc_optimizer_units']['mode'],actual_native_L_contexts=m._feature_l_steps,synthetic_joint_optimizer_steps=16,actual_optimizer_group_count=len(opt.param_groups),original_U_loss_object_unchanged=True,selected_Sinc_snapshot_recount=True,default_W_gradient_nonzero=float(m.id_backbone.encoder.core.id_backbone.cls_head.weight.grad.norm())>0,source_or_target_IQ_read=False,head_fit_updates=0))
 assert all(r['actual_native_L_contexts']==16 and r['default_W_gradient_nonzero'] for r in actual)
 public=json.loads(json.dumps([d.config(r) for r in d.rows()]).replace(d.PROJECT,'PROJECT'))
 from experiments.cvs_temporal_rsc50.dispatch import restore_project
 assert restore_project(public,d.PROJECT)==[d.config(r) for r in d.rows()]
 q=dict(status='PASS',source_only=True,target_metrics_read_for_decision=False,fixed_rules=d.first.config(d.first.training_rows()[0])['source_falsification'],predefined_remaining_seeds=list(d.SEEDS),head_fit_updates=0,last5_mean_L_CE={r['arm']:.2 for r in d.first.training_rows()},unit_changed_cutoffs=24,unit_changed_bandwidths=24,unit_student_kernel_relative_L2=1e-5,unit_ownEMA_kernel_relative_L2=1e-5,actual_candidate_minus_control_L720zL2=1e-5,actual_observations=[dict(status='VERIFIED',row_id=r['row_id'],actual_steps=11100,actual_U_packets=963900,actual_Sinc_parameter_records=104,actual_snapshot_batches=52,head_fit_updates=0,extra_model_forwards=0,U_truth_read=False,query_read=False,source_only=True) for r in d.first.training_rows()])
 d.validate_qualification(q)
 bads=[]
 for key,value in [('status','FAIL'),('source_only',False),('target_metrics_read_for_decision',True),('fixed_rules',{}),('predefined_remaining_seeds',[2026092702]),('head_fit_updates',1),('unit_changed_cutoffs',0),('unit_ownEMA_kernel_relative_L2',float('nan')),('actual_observations',[])]:
  bad=copy.deepcopy(q);bad[key]=value;bads.append(bad)
 bad=copy.deepcopy(q);bad['actual_observations'][0]['actual_steps']=11099;bads.append(bad)
 bad=copy.deepcopy(q);bad['last5_mean_L_CE']['SIG_TIME']=.6;bads.append(bad)
 for bad in bads:
  try:d.validate_qualification(bad)
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid original source qualification accepted')
 from .publish import configure,transport
 configure();compile(transport.REMOTE,'<sinc-seeds-remote>','exec');compile(transport.INSPECT,'<sinc-seeds-inspect>','exec')
 assert 'experiments.cvs_sinc_units_seeds.preflight' in transport.REMOTE and 'experiments.cvs_sinc_units_seeds.dispatch' in transport.REMOTE
 assert all(p in transport.PATHS for p in ('experiments/cvs_sinc_units50','experiments/cvs_sigtime_pair_seeds','experiments/cvs_sinc_units_seeds','experiments/cvs_source_neighbor_pairs','experiments/cvs_source_gate_audit'))
 return dict(status='PASS',negatives=negatives,native_difference_keys=native_changes,full_canonical_roundtrip=True,first_source_qualification_read=False,unchanged_first_candidate_tests_repeated=False,scientific_dictionary_identical=True,actual_rows=actual,same_seed_initial_state_equal=True,actual_native_arg_differences=diff,global_optimizer_class_restored=True,no_new_feature_loss_or_forward=True,real_IQ_or_checkpoint_read=False,formal_training_started=False)
if __name__=='__main__':print(json.dumps(checks()))
