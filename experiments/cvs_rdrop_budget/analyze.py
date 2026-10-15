from pathlib import Path
import sys,json
W=Path(__file__).resolve().parents[2];sys.path[:0]=[str(W),str(W/'code')]
from experiments.cvs_rdrop_budget import design as d
from experiments.cvs_phase1_stack.publish import ssh
from experiments.cvs_rdrop_budget.publish import PYTHON
P=W/'local_artifacts'/d.RELEASE
v=d.read(P/'independent_readback.json')
if not v['completion'] or v['dispatcher'] or any(r['process'] for r in v['rows']):print('NOT_TERMINAL; no artifact analysis or truth reads started');sys.exit(0)
if (P/'independent_analysis.json').is_file():print('ALREADY_ANALYZED');sys.exit(0)
assert v['completion']['status']=='ARTIFACTS_COMPLETE' and not v['failure'] and all(not any(r['fatal_markers'].values()) and r['smoke_PASS_count']==1 for r in v['rows'])
script=r'''
import sys,types,json,gzip,math,gc,re,statistics,collections,time
from pathlib import Path
import torch,numpy as np
c=CONFIG
original,fold,release=(Path(c[k]) for k in ('dependency','fold','release'))
sys.path[:0]=[str(original),str(original/'code'),str(original/'experiments/adv3b02_xuc/code')]
for suffix,root in (('cvs_rdrop_budget',release),('cvs_rdrop_fold20',release),('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold)):
 name='experiments.'+suffix;p=types.ModuleType(name);p.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=p
from experiments.cvs_rdrop_budget import design as d,model as api
from experiments.cvs_rdrop_budget.provenance import source_proof,validate_checkpoint
from experiments.cvs_rdrop_budget.evaluate import preflight
from experiments.cvs_report_ssl.evaluate import truth_arrays
from experiments.cvs_feature_sequence50.features import SequenceReference
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_equivariant_identity.precision import numerical_context
torch.set_num_threads(2)
execution=(release/'release_commit.txt').read_text().strip();matrix=d.read(d.BASE/'result.json');completion=d.read(d.BASE/'completion.json')
assert matrix['status']==completion['status']=='ARTIFACTS_COMPLETE' and matrix['run_id']==d.RUN and completion['rows']==12 and completion['metric_records']==216
# Verify every row's frozen prediction BEFORE the independent truth-side read.
proofs={};physical=None
for row in d.rows():
 rid=row['row_id'];config,ids=preflight(rid);proofs[rid]=source_proof(config)
 assert d.read(d.BASE/rid/'completion.json')['status']=='MODEL_PREDICTED_SCORED'
 if physical is None:physical=ids
 else:assert np.array_equal(physical,ids)
y,rx,days=truth_arrays(d.read(d.TRUTH),physical)
results={};saved={};initials={};recounted=[]
for row in d.rows():
 rid=row['row_id'];config=d.config(row);out=proofs[rid];e=config['epochs'];steps=e*222;label=config['label_epochs'];pseudo=config['pseudo_epochs']
 done=d.read(out/'completion.json');resolved=d.read(out/'resolved_config.json');args=d.read(out/'actual_native_args.json');contract=d.read(out/'source_contract.json')
 assert resolved['commit']==execution and {k:len(v) for k,v in contract['role_ids'].items()}=={'L_s':6300,'U_s':56700,'V':27000}
 records=[json.loads(line) for line in gzip.open(out/'step_metrics.jsonl.gz','rt') if line.strip()]
 epochs=[json.loads(line) for line in (out/'epoch_metrics.jsonl').read_text().splitlines() if line.strip()]
 assert len(records)==steps and [x['step'] for x in records]==list(range(1,steps+1)) and len(epochs)==e and [x['epoch'] for x in epochs]==list(range(1,e+1))
 keys=('train/loss','train/loss_tx_labeled','train/loss_unlabeled','train/loss_closed_group','train/loss_open_group','rdrop/CE1','rdrop/CE2','rdrop/paired_CE','rdrop/symmetric_KL')
 values={k:[] for k in keys};epoch_values={};pseudo_total=pseudo_selected=0;objective_error=0.;alpha=config['training_mechanism']['alpha']
 for step_row in records:
  step=step_row['step'];m=step_row['metrics'];epoch=(step-1)//222+1;assert step_row['epoch']==epoch
  assert m['rdrop/calls']==step and m['rdrop/extra_L_packets']==step*128 and m['rdrop/alpha']==alpha and float(m.get('train/pseudo_truth_available',0) or 0)==0
  for key in keys:
   value=m[key];assert isinstance(value,(float,int)) and math.isfinite(value);values[key].append(value);epoch_values.setdefault(epoch,{}).setdefault(key,[]).append(value)
  assert abs(m['rdrop/paired_CE']-.5*(m['rdrop/CE1']+m['rdrop/CE2']))<=1e-6 and abs(m['train/loss_tx_labeled']-m['rdrop/CE1'])<=1e-6
  assert m['train/tail_rollback_closed_scale']==1 and m['train/loss_open_group']==0
  # Native logging does not independently expose entropy. This is a derived
  # weighted residual, explicitly distinguished from an observed entropy field.
  entropy=m['train/loss_closed_group']-m['train/loss_tx_labeled']-args['lambda_u']*m['train/loss_unlabeled']
  assert entropy>=-1e-6 and (epoch>label or abs(entropy)<=1e-6)
  key='derived/original_weighted_U_entropy';values.setdefault(key,[]).append(entropy);epoch_values[epoch].setdefault(key,[]).append(entropy)
  expected=m['train/loss_closed_group']+.5*(m['rdrop/CE2']-m['rdrop/CE1'])+alpha*m['rdrop/symmetric_KL'];objective_error=max(objective_error,abs(m['train/loss']-expected))
  total=int(m['train/pseudo_total']) if epoch>label else 0;selected=int(m['train/pseudo_selected']) if epoch>label else 0
  assert 0<=selected<=total
  pseudo_total+=total;pseudo_selected+=selected
  epoch_values[epoch].setdefault('observed/U_total',[]).append(total);epoch_values[epoch].setdefault('observed/U_selected',[]).append(selected)
 assert pseudo_total==pseudo*56700 and objective_error<=2e-5
 assert abs(sum(values['train/loss_tx_labeled'][-222:])/222-done['final_training_L_CE'])<=1e-8
 stdout=(d.BASE/'logs'/(rid+'-train_predict.log')).read_text(errors='replace')
 fatal={key:stdout.count(key) for key in ('Traceback (most recent call last)','CUDA out of memory','FloatingPointError:','ValueError:','RuntimeError:','TypeError:')};assert not sum(fatal.values()) and stdout.count('SMOKE PASS own scratch checkpoint; query_read=false')==1 and 'PREDICTIONS_COMPLETE '+rid in stdout
 warnings=collections.Counter(line.strip() for line in stdout.splitlines() if 'warning' in line.lower() or 'warn:' in line.lower())
 # Metadata/provenance was already checked for this row before deserialization.
 cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(cp,config)
 initial_payload=torch.load(out/'initial_smoke.pt',map_location='cpu',weights_only=False);initial=initial_payload['model'];assert initial_payload['scratch_only']
 with numerical_context(d.FULL_FP32_POLICY):
  model=api.build(config,'cpu').eval().requires_grad_(False);named=dict(model.named_parameters());Wname='id_backbone.encoder.core.id_backbone.cls_head.weight'
  feature_names=[n for n in named if n.startswith('id_backbone.encoder.') and n!=Wname];assert cp['model'][Wname].shape==(6,160)
  feature_delta=sum(float((cp['model'][n].double()-initial[n].double()).square().sum()) for n in feature_names)**.5;Wdelta=float((cp['model'][Wname].double()-initial[Wname].double()).norm());assert feature_delta>0 and Wdelta>0
  torch.manual_seed(2026101542);x=torch.randn(2,2,256)
  def exported():
   with torch.no_grad(),fresh_parameter_caches(model):return model.forward_identity_only(x)['id_feat_joint'].clone()
  model.load_state_dict(initial,strict=True);before=exported();model.load_state_dict(cp['model'],strict=True);after=exported();ref=next(m for m in model.modules() if isinstance(m,SequenceReference))
  with torch.no_grad(),fresh_parameter_caches(model):direct=ref.features(x).clone()
  weight=api.classifier(model).weight;old=weight.detach().clone()
  with torch.no_grad():weight.add_(torch.linspace(-.3,.4,weight.numel()).reshape_as(weight))
  perturbed=exported()
  with torch.no_grad():weight.copy_(old)
  assert torch.isfinite(before).all() and torch.isfinite(after).all() and (after-before).norm()>1e-8 and torch.equal(after,direct) and torch.equal(after,perturbed)
 with np.load(d.BASE/rid/'prediction/predictions.npz',allow_pickle=False) as p:
  assert np.array_equal(p['ids'],physical);pred=p['NATIVE_HEAD'].copy()
 ok=pred==y
 with np.load(d.BASE/rid/'scores/correctness.npz',allow_pickle=False) as p:assert np.array_equal(p['ids'],physical) and np.array_equal(p['NATIVE_HEAD'],ok)
 metric_records=d.read(d.BASE/rid/'scores/metrics.json')['records'];assert len(metric_records)==18
 for record in metric_records:
  dim,s=record['dimension'],record['stratum']
  mask=np.ones(len(y),bool) if dim=='overall' else rx==s if dim=='receiver' else days==s if dim=='day' else y==d.CLASSES.index(s)
  cm=np.bincount(6*y[mask]+pred[mask],minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1)
  acc=float(np.trace(cm)/cm.sum());f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean())
  assert record['row_id']==rid and record['arm']==config['arm'] and record['model_seed']==config['model_seed'] and record['view']=='clean' and record['readout']=='NATIVE_HEAD'
  assert record['query_count']==int(mask.sum()) and record['confusion']==cm.tolist() and abs(record['accuracy']-acc)<=1e-12 and abs(record['macro_f1']-f1)<=1e-12
  recounted.append(record)
 stats={k:dict(count=len(v),mean=statistics.mean(v),minimum=min(v),maximum=max(v)) for k,v in values.items()}
 curves=[dict(epoch=n,label_stage=n<=label,epoch_time_s=epochs[n-1]['epoch_time_s'],lr=epochs[n-1]['lr'],source_V_accuracy_percent=epochs[n-1]['val_tx_acc'],**{k:statistics.mean(v) for k,v in group.items()}) for n,group in epoch_values.items()]
 results[rid]=dict(row_id=rid,arm=config['arm'],status='VERIFIED',model_seed=config['model_seed'],epochs=e,label_epochs=label,pseudo_epochs=pseudo,optimizer_steps=steps,extra_L_identity_forwards=steps*128,pseudo_observations=pseudo_total,pseudo_selected_observations=pseudo_selected,final_training_L_CE=done['final_training_L_CE'],source_health=done['final_training_L_CE']<=.5,source_V=done['final_source_metrics'],objective_max_abs_error=objective_error,complete_step_loss_stats=stats,complete_epoch_curves=curves,feature_parameter_tensors=len(feature_names),feature_parameter_L2_delta=feature_delta,default_W_L2_delta=Wdelta,synthetic_boundary=dict(status='PASS',feature_initial_final_L2=float((after-before).double().norm()),feature_only_matches_joint160_bitwise=True,perturbing_W_leaves_joint160_bitwise=True,synthetic_packet_forwards=8,actual_IQ_reads=0,optimizer_steps=0,head_fit_updates=0,checkpoint_mutations=0),elapsed_seconds=done['elapsed_seconds'],prediction_seconds=d.read(d.BASE/rid/'prediction/complete.json')['wall_seconds'],stdout_bytes=len(stdout.encode()),stdout_lines=len(stdout.splitlines()),full_stdout_fatal_counts=fatal,full_stdout_warning_counts=dict(warnings),head_fit_updates=0,U_truth_reads=0)
 initials[rid]=initial;saved[rid]=ok;del model,cp,records;gc.collect()
matched={}
for seed in d.SEEDS:
 group=[initials[r['row_id']] for r in d.rows() if r['model_seed']==seed];base=group[0]
 assert all(set(base)==set(other) and all(torch.equal(base[k],other[k]) for k in base) for other in group[1:]);matched[str(seed)]=len(base)
comparisons=(('E20_RDROP','E20_TWO_CE'),('E50_RDROP','E50_TWO_CE'),('E50_TWO_CE','E20_TWO_CE'),('E50_RDROP','E20_RDROP'))
pairs=[];stratum_pairs=[];paired={}
for after,before in comparisons:
 key=after+' minus '+before
 for seed in d.SEEDS:
  b,a=(saved[arm+'-s'+str(seed)] for arm in (before,after));repair=int((~b&a).sum());broken=int((b&~a).sum())
  pairs.append(dict(comparison=key,model_seed=seed,corrected_errors=repair,introduced_errors=broken,accuracy_difference_pp=(repair-broken)/1680))
  for dim,strata in (('receiver',sorted(set(rx))),('transmitter',d.CLASSES),('day',sorted(set(days)))):
   for s in strata:
    mask=rx==s if dim=='receiver' else days==s if dim=='day' else y==d.CLASSES.index(s)
    correct=int((~b[mask]&a[mask]).sum());new=int((b[mask]&~a[mask]).sum());n=int(mask.sum())
    stratum_pairs.append(dict(comparison=key,model_seed=seed,dimension=dim,stratum=s,count=n,corrected=correct,introduced=new,accuracy_difference_pp=(correct-new)*100/n))
 values=[p['accuracy_difference_pp'] for p in pairs if p['comparison']==key];paired[key]=dict(seed_count=3,mean_pp=statistics.mean(values),sample_SD_pp=statistics.stdev(values),positive_seeds=sum(v>0 for v in values))
summaries={}
for arm in d.ARMS:
 group=[r for r in recounted if r['arm']==arm and r['dimension']=='overall'];assert len(group)==3 and {r['model_seed'] for r in group}==set(d.SEEDS)
 details=[]
 for r in group:
  strata=[x for x in recounted if x['row_id']==r['row_id']]
  details.append(dict(model_seed=r['model_seed'],accuracy=r['accuracy'],macro_f1=r['macro_f1'],errors=r['query_count']-sum(r['confusion'][i][i] for i in range(6)),worst_RX_accuracy=min(x['accuracy'] for x in strata if x['dimension']=='receiver'),worst_day_accuracy=min(x['accuracy'] for x in strata if x['dimension']=='day')))
 summaries[arm]=dict(seed_count=3,seeds=sorted(d.SEEDS),per_seed=details,**{k+'_mean':statistics.mean(r[k] for r in details) for k in ('accuracy','macro_f1','worst_RX_accuracy','worst_day_accuracy')},accuracy_sample_SD=statistics.stdev(r['accuracy'] for r in details))
assert matrix['records']==recounted and matrix['paired_errors']==pairs and matrix['paired_summary']==paired and matrix['summary']==summaries and matrix['totals']==d.totals()
assert matrix['primary']=='E50_RDROP' and not matrix['goal_completed'] and matrix['E50_RDROP_goal84_check']==(summaries['E50_RDROP']['accuracy_mean']>=.84) and matrix['E20_RDROP_goal84_check']==(summaries['E20_RDROP']['accuracy_mean']>=.84)
launch=d.read(d.BASE/'launch.json')['rows'];finish_time=(d.BASE/'completion.json').stat().st_mtime;matrix_seconds=finish_time-min(r['read_at'] for r in launch)
result=dict(status='VERIFIED',run_id=d.RUN,execution_commit=execution,rows=results,initial_state_equal_count_by_seed=matched,summary=summaries,records=recounted,paired_errors=pairs,paired_summary=paired,stratum_pairs=stratum_pairs,totals=d.totals(),primary='E50_RDROP',E50_RDROP_goal84_check=matrix['E50_RDROP_goal84_check'],E20_RDROP_goal84_check=matrix['E20_RDROP_goal84_check'],actual_IQ_reads=0,new_target_predictions=0,existing_target_predictions_recounted=2016000,independent_truth_last_recount=True,source_truth_reads=0,U_truth_reads=0,head_fit_updates=0,analysis_optimizer_steps=0,analysis_checkpoint_mutations=0,separate_synthetic_boundary_forwards=96,matrix_elapsed_seconds=matrix_seconds,aggregate_completed_rows_per_hour=12*3600/matrix_seconds,goal_completed=False)
print(json.dumps(result))
'''
settings=dict(dependency=d.DEPENDENCY_RELEASE,fold=d.FOLD_RELEASE,release=d.PROJECT+'/releases/'+d.RELEASE)
payload=script.replace('CONFIG',repr(settings));compile(payload,'rdrop-budget-independent-analysis','exec')
outer='import subprocess,os\nr=subprocess.run('+repr([PYTHON,'-c',payload])+',capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"))\nif r.returncode:raise RuntimeError(r.stderr)\nprint(r.stdout)'
result=json.loads(ssh(outer));d.write(P/'independent_analysis.json',result)
print(json.dumps({k:result[k] for k in ('status','run_id','summary','paired_summary','initial_state_equal_count_by_seed','E50_RDROP_goal84_check','E20_RDROP_goal84_check','matrix_elapsed_seconds')},ensure_ascii=False))
