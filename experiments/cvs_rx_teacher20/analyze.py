"""Read-only full source artifact analysis; no IQ, optimizer, or target access."""
from pathlib import Path
import json,sys
W=Path(__file__).resolve().parents[2];sys.path[:0]=[str(W),str(W/'code')]
from experiments.cvs_rx_teacher20 import design as d
from experiments.cvs_rx_teacher20.publish import PYTHON
from experiments.cvs_phase1_stack.publish import ssh
P=W/'local_artifacts'/d.RELEASE
v=d.read(P/'independent_readback.json')
if not v['completion'] or v['dispatcher'] or v['compare_process'] or any(r['process'] for r in v['rows']):
 raise RuntimeError('Reconcile same-run terminal state before source artifact analysis')
if (P/'independent_analysis.json').is_file():
 print('ALREADY_ANALYZED');sys.exit(0)
assert v['completion']['status']=='SOURCE_MATRIX_COMPLETE' and not v['failure']
assert all(r['smoke_PASS_count']==1 and not any(r['fatal_counts'].values()) for r in v['rows'])
script=r'''
import sys,types,json,gzip,math,gc,statistics,collections,importlib.util
from pathlib import Path
import torch,numpy as np
c=CONFIG
original,fold,release=(Path(c[k]) for k in ('dependency','fold','release'))
sys.path[:0]=[str(original),str(original/'code'),str(original/'experiments/adv3b02_xuc/code')]
for suffix,root in (('cvs_rx_teacher20',release),('cvs_rdrop_fold20',release),('cvs_evidence_compensation20',release),('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold)):
 name='experiments.'+suffix;p=types.ModuleType(name);p.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=p
name='experiments.cvs_feature_sequence50.source';spec=importlib.util.spec_from_file_location(name,fold/'experiments/cvs_feature_sequence50/source.py');m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
from experiments.cvs_rx_teacher20 import design as d,model as api
from experiments.cvs_rx_teacher20.provenance import source_proof,validate_checkpoint
from experiments.cvs_rx_teacher20.compare import readout,qualify
from experiments.cvs_feature_sequence50.features import SequenceReference
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_evidence_responsibility_train20 import design as a0
from experiments.cvs_evidence_responsibility_train20.provenance import source_proof as a0_proof
torch.set_num_threads(2)
execution=(release/'release_commit.txt').read_text().strip();matrix=d.read(d.BASE/'matrix_result.json');complete=d.read(d.BASE/'completion.json')
assert matrix['run_id']==d.RUN and matrix['status']==complete['status']=='SOURCE_MATRIX_COMPLETE' and complete['rows']==2
assert matrix['actual_epochs']==40 and matrix['actual_optimizer_steps']==8880 and matrix['source_only'] and not matrix['target_access'] and not matrix['U_truth_read'] and matrix['head_fit_updates']==0
proofs={r['row_id']:source_proof(d.config(r)) for r in d.rows()}
readouts={r['arm']:readout(d.config(r)) for r in d.rows()}
a,b=(readouts[arm] for arm in d.ARMS)
assert a['physical_ids']==b['physical_ids'] and a['labels']==b['labels']
proxy=d.read(d.BASE/'A0_gate_proxy.json');ids=a['physical_ids'];y=np.asarray(a['labels']);pa=np.asarray(a['predictions']);pb=np.asarray(b['predictions']);base=pa==y;candidate=pb==y
a0_c=a0.config(next(r for r in a0.rows() if r['arm']=='BASE_SSL' and r['model_seed']==d.SEED));a0_proof(a0_c)
assert proxy['status']=='SOURCE_A0_GATE_PROXY_COMPLETE' and proxy['A0_run_id']==a0_c['run_id'] and proxy['A0_row_id']==a0_c['row_id'] and proxy['A0_epoch']==20
assert proxy['physical_ids']==ids and proxy['labels']==a['labels'] and proxy['count']==5400
assert proxy['order_seed']==2026101502 and proxy['view_seed']==2026101503 and proxy['batch_size']==256 and proxy['student_strong_std']==.015 and proxy['student_dropout_active']
assert proxy['parameters_buffers_gradients_unchanged'] and proxy['metadata_TX_read_by_gate'] is False and proxy['target_access'] is False
assert all(proxy[k]==0 for k in ('optimizer_steps','EMA_updates','head_fit_updates','U_IQ_reads','U_truth_reads'))
accepted=np.asarray(proxy['accepted']);structural=np.asarray(proxy['structural']);conf=np.asarray(proxy['confidence']);a0_pred=np.asarray(proxy['original_EMA_pseudo'])
assert accepted.shape==structural.shape==conf.shape==a0_pred.shape==(5400,) and accepted.dtype==structural.dtype==bool and np.isfinite(conf).all() and ((conf>=0)&(conf<=1)).all() and set(a0_pred)<=set(range(6)) and not (accepted&~structural).any()
rejected=~accepted;days=np.asarray([i[2] for i in ids]);tp={arm:np.exp(np.asarray(readouts[arm]['log_probabilities'])[np.arange(5400),y]) for arm in d.ARMS};a0_ok=a0_pred==y
records=[]
for dim,strata in (('all',['ALL']),('gate',['accepted','rejected']),('day',[1,2,3]),('TX',list(range(6)))):
 for value in strata:
  mask=np.ones(5400,bool) if dim=='all' else (accepted if value=='accepted' else rejected) if dim=='gate' else days==value if dim=='day' else y==value
  n=int(mask.sum());repair=int((~base&candidate&mask).sum());broken=int((base&~candidate&mask).sum())
  records.append(dict(dimension=dim,stratum=value,count=n,SAMPLE_80_accuracy=float(base[mask].mean()) if n else None,EXCLUDE_RX1_accuracy=float(candidate[mask].mean()) if n else None,SAMPLE_80_mean_true_probability=float(tp['SAMPLE_80'][mask].mean()) if n else None,EXCLUDE_RX1_mean_true_probability=float(tp['EXCLUDE_RX1'][mask].mean()) if n else None,A0_EMA_accuracy=float(a0_ok[mask].mean()) if n else None,corrected=repair,introduced=broken,accuracy_difference_pp=(repair-broken)*100/n if n else None))
class_pairs=[dict(true_TX=t,wrong_TX=w,SAMPLE_80_errors=int(((y==t)&(pa==w)).sum()),EXCLUDE_RX1_errors=int(((y==t)&(pb==w)).sum()),corrected=int(((y==t)&(pa==w)&candidate).sum()),introduced=int(((y==t)&base&(pb==w)).sum())) for t in range(6) for w in range(6) if t!=w]
qualification=qualify(a,b,rejected)
predicate=bool(a['source_health'] and b['source_health'] and b['errors']<a['errors'] and b['mean_CE']<=a['mean_CE'] and rejected.any() and records[2]['corrected']>records[2]['introduced'])
assert matrix['records']==records and matrix['class_pair_records']==class_pairs and matrix['qualification']==qualification and qualification['qualified']==complete['qualified']==predicate
rows={};initials={}
for row in d.rows():
 rid=row['row_id'];config=d.config(row);out=proofs[rid];done=d.read(out/'completion.json');resolved=d.read(out/'resolved_config.json')
 assert resolved['commit']==execution
 steps=[json.loads(s) for s in gzip.open(out/'step_metrics.jsonl.gz','rt') if s.strip()]
 epochs=[json.loads(s) for s in (out/'epoch_metrics.jsonl').read_text().splitlines() if s.strip()]
 assert len(steps)==4440 and [s['step'] for s in steps]==list(range(1,4441)) and len(epochs)==20 and [s['epoch'] for s in epochs]==list(range(1,21))
 keys=('train/loss','train/loss_tx_labeled','train/tx_acc','train/grad_before_clip','train/grad_total');values={k:[] for k in keys};groups={};max_objective_error=0
 for s in steps:
  n=s['step'];epoch=(n-1)//222+1;m=s['metrics'];assert s['epoch']==epoch and m['teacher/verified_L_steps']==n and 0<m['teacher/unique_L_packets']<=5040 and m['train/optimizer_step_applied']==1
  assert all(not isinstance(value,(float,int)) or math.isfinite(value) for value in m.values())
  assert m['train/skipped_nonfinite_loss']==m['train/skipped_nonfinite_grad']==m['train/pseudo_total']==m['train/pseudo_selected']==m['train/pseudo_truth_available']==0
  assert m['train/loss_unlabeled']==m['train/loss_open_group']==m['train/concat_sat_applied']==0
  max_objective_error=max(max_objective_error,abs(m['train/loss']-m['train/loss_tx_labeled']),abs(m['train/loss_closed_group']-m['train/loss_tx_labeled']))
  for k in keys:values[k].append(m[k]);groups.setdefault(epoch,{}).setdefault(k,[]).append(m[k])
 assert max_objective_error<=1e-7 and steps[-1]['metrics']['teacher/unique_L_packets']==5040
 assert abs(statistics.mean(values['train/loss_tx_labeled'][-222:])-done['final_training_L_CE'])<=1e-9
 curves=[]
 for e in epochs:
  n=e['epoch'];assert e['test_tx_total']==0 and e['test_tx_acc'] is None and e['val_tx_total']==21600 and e['train_optimizer_step_applied']==1
  assert abs(statistics.mean(groups[n]['train/loss_tx_labeled'])-e['train_loss'])<=1e-8
  curves.append(dict(epoch=n,epoch_time_s=e['epoch_time_s'],lr=e['lr'],source_V_accuracy_percent=e['val_tx_acc'],**{k:statistics.mean(v) for k,v in groups[n].items()}))
 stdout=(d.BASE/'logs'/(rid+'.log')).read_text(errors='replace')
 fatal={key:stdout.count(key) for key in ('Traceback (most recent call last)','CUDA out of memory','ValueError:','RuntimeError:','FloatingPointError:','TypeError:')}
 assert not sum(fatal.values()) and stdout.count('SMOKE PASS own scratch checkpoint; query_read=false')==1 and 'SOURCE_TEACHER_COMPLETE '+rid in stdout
 warnings=collections.Counter(line.strip() for line in stdout.splitlines() if 'warning' in line.lower())
 cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(cp,config)
 payload=torch.load(out/'initial_smoke.pt',map_location='cpu',weights_only=False);initial=payload['model'];assert payload['scratch_only']
 with numerical_context(d.FULL_FP32_POLICY):
  model=api.build(config,'cpu').eval().requires_grad_(False);named=dict(model.named_parameters());Wname='id_backbone.encoder.core.id_backbone.cls_head.weight'
  feature_names=[name for name in named if name.startswith('id_backbone.encoder.') and name!=Wname]
  fd=sum(float((cp['model'][name].double()-initial[name].double()).square().sum()) for name in feature_names)**.5;wd=float((cp['model'][Wname].double()-initial[Wname].double()).norm())
  assert cp['model'][Wname].shape==(6,160) and fd>0 and wd>0
  torch.manual_seed(2026101542);x=torch.randn(2,2,256)
  def features():
   with torch.no_grad(),fresh_parameter_caches(model):return model.forward_identity_only(x)['id_feat_joint'].clone()
  model.load_state_dict(initial,strict=True);before=features();model.load_state_dict(cp['model'],strict=True);after=features();ref=next(m for m in model.modules() if isinstance(m,SequenceReference))
  with torch.no_grad(),fresh_parameter_caches(model):direct=ref.features(x).clone()
  weight=api.classifier(model).weight;old=weight.detach().clone()
  with torch.no_grad():weight.add_(torch.linspace(-.3,.4,weight.numel()).reshape_as(weight))
  perturbed=features()
  with torch.no_grad():weight.copy_(old)
  assert torch.isfinite(before).all() and torch.isfinite(after).all() and (after-before).norm()>1e-8 and torch.equal(after,direct) and torch.equal(after,perturbed)
 held=readouts[row['arm']];subset=d.read(out/'teacher_subset.json');train=subset['train_records']
 rows[rid]=dict(row_id=rid,arm=row['arm'],model_seed=d.SEED,status='VERIFIED',epochs=20,optimizer_steps=4440,train_L_unique=5040,source_role_counts={'L_s':5040,'U_s':0,'V':21600},train_RX=config['teacher_exposure']['train_RX'],per_RX_TX_day=config['teacher_exposure']['per_RX_TX_day'],RX1_train_L=sum(r['RX']==1 for r in train),seen_L_packets=5040,U_iterator_calls=0,EMA=False,head_fit_updates=0,checkpoint_sources=[],ancestors=[],final_training_L_CE=done['final_training_L_CE'],source_V=done['final_source_metrics'],held_RX1={key:held[key] for key in ('accuracy','errors','mean_CE','mean_true_probability','mean_max_probability_on_errors','count','source_health')},objective_max_abs_error=max_objective_error,complete_step_loss_stats={k:dict(count=len(v),mean=statistics.mean(v),minimum=min(v),maximum=max(v)) for k,v in values.items()},complete_epoch_curves=curves,elapsed_seconds=done['elapsed_seconds'],feature_parameter_L2_delta=fd,default_W_L2_delta=wd,synthetic_boundary=dict(status='PASS',feature_initial_final_L2=float((after-before).double().norm()),feature_only_matches_joint160_bitwise=True,perturbing_W_leaves_joint160_bitwise=True,synthetic_packet_forwards=8),stdout_bytes=len(stdout.encode()),stdout_lines=len(stdout.splitlines()),full_stdout_fatal_counts=fatal,full_stdout_warning_counts=dict(warnings))
 initials[rid]=initial;del steps,model,cp;gc.collect()
ia,ib=initials.values();assert set(ia)==set(ib) and all(torch.equal(ia[key],ib[key]) for key in ia)
compare_log=(d.BASE/'logs/source_compare.log').read_text(errors='replace');assert 'SOURCE_TEACHER_COMPARISON_COMPLETE' in compare_log and 'Traceback (most recent call last)' not in compare_log
launch=d.read(d.BASE/'launch.json')['rows'];wall=(d.BASE/'completion.json').stat().st_mtime-min(r['read_at'] for r in launch)
result=dict(status='VERIFIED',run_id=d.RUN,execution_commit=execution,rows=rows,qualification=qualification,qualification_recomputed=True,records=records,class_pair_records=class_pairs,initial_equal_state_tensors=len(ia),A0_source_proxy=dict(accepted=int(accepted.sum()),rejected=int(rejected.sum()),structural=int(structural.sum()),EMA_errors=int((~a0_ok).sum()),EMA_accuracy=float(a0_ok.mean()),original_source_provenance_verified=True,metadata_TX_read_by_gate=False,parameters_buffers_gradients_unchanged=True,limitation=proxy['limitation']),matrix_elapsed_seconds=wall,complete_steps=8880,complete_epochs=40,teacher_train_L_forward_packets=1136640,held_teacher_source_V_decisions_recounted=10800,A0_source_V_gate_decisions_recounted=5400,separate_synthetic_boundary_forwards=16,analysis_actual_IQ_reads=0,analysis_optimizer_steps=0,analysis_head_fit_updates=0,analysis_checkpoint_mutations=0,U_truth_reads=0,target_access=False,teacher_bank_or_student_launched=False,goal_completed=False)
print(json.dumps(result))
'''
settings=dict(dependency=d.DEPENDENCY_RELEASE,fold=d.FOLD_RELEASE,release=d.PROJECT+'/releases/'+d.RELEASE)
payload=script.replace('CONFIG',repr(settings));compile(payload,'teacher-source-independent-analysis','exec')
outer='import subprocess,os\nr=subprocess.run('+repr([PYTHON,'-c',payload])+',capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"))\nif r.returncode:raise RuntimeError(r.stderr)\nprint(r.stdout)'
result=json.loads(ssh(outer));d.write(P/'independent_analysis.json',result)
print(json.dumps({k:result[k] for k in ('status','run_id','qualification','initial_equal_state_tensors','matrix_elapsed_seconds','complete_steps','complete_epochs')},ensure_ascii=False))
