"""Read-only complete-log and frozen-prediction analysis of the fixed pure-F run.

The remote SCRIPT performs zero model forwards, optimizer updates, predictions,
or scoring jobs. It recounts saved decisions only after completion and the
source decision. Model-seed n=1 is kept explicit throughout.
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = '20261019-phase1-F-rawAs-e20-s2026092701-m2-r01'
REFERENCE_RUN = '20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01'
RELEASE = 'cvs_F_rawAs_E20_20261019_r01'
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')

SCRIPT = r'''
import csv,gzip,importlib.util,itertools,json,math,re
from pathlib import Path
import numpy as np

def read(p):
 return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def records(p):
 p=Path(p)
 with (gzip.open(p,'rt',encoding='utf8') if p.suffix=='.gz' else p.open(encoding='utf8')) as f:
  return [json.loads(s) for s in f if s.strip()]

def scalar(record,key):
 value=record[key]
 if value is None or not isinstance(value,(int,float)) or not math.isfinite(value):
  raise ValueError('Missing/nonfinite required scalar '+key)
 return float(value)

def confusion_metrics(y,predicted):
 cm=np.bincount(y*6+predicted,minlength=36).reshape(6,6)
 den=cm.sum(0)+cm.sum(1)
 return dict(n=len(y),accuracy=float(np.trace(cm)/len(y)),
             macro_f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean()),
             confusion=cm.tolist())

def paired(y,before,after):
 b=before==y;m=after==y;both=int((b&m).sum());broken=int((b&~m).sum())
 return dict(n=len(y),corrected=int((~b&m).sum()),damaged=broken,
             both_correct=both,both_wrong=int((~b&~m).sum()),
             net_pp=float(100*(m.mean()-b.mean())),
             baseline_correct_retention=None if both+broken==0 else both/(both+broken),
             model_seed_count=1,sample_SD=None,positive_model_seeds=int(m.mean()>b.mean()))

def sequence_counts(path):
 counts=dict(L=0,U=0);packets=dict(L=0,U=0)
 with gzip.open(path,'rt',encoding='utf8') as stream:
  for line in stream:
   r=json.loads(line);role=r['role'];counts[role]+=1;packets[role]+=len(r['ids'])
 if counts!={'L':4440,'U':1554} or packets!={'L':568320,'U':396900}:
  raise ValueError('Original complete L/U sequence budget differs')
 return dict(batches=counts,packets=packets)

def sequence_pairs(paths):
 arms=list(paths);comparisons=list(itertools.combinations(arms,2))
 result={a+'__'+b:dict(equal=True,first_mismatch_line=None,
          by_role={r:dict(equal=True,first_mismatch_batch=None,compared_batches=0) for r in ('L','U')})
         for a,b in comparisons}
 streams=[gzip.open(paths[a]/'source/training_sequence.jsonl.gz','rt',encoding='utf8') for a in arms]
 try:
  for index,items in enumerate(itertools.zip_longest(*streams),1):
   parsed=[None if s is None else json.loads(s) for s in items]
   for a,b in comparisons:
    i,j=arms.index(a),arms.index(b);entry=result[a+'__'+b];left,right=parsed[i],parsed[j]
    equal=left==right
    if not equal:
     entry['equal']=False
     if entry['first_mismatch_line'] is None:entry['first_mismatch_line']=index
    if left is not None and right is not None and left['role']==right['role']:
     row=entry['by_role'][left['role']];row['compared_batches']+=1
     if not equal:
      row['equal']=False
      if row['first_mismatch_batch'] is None:row['first_mismatch_batch']=row['compared_batches']
    elif not equal:
     for role in ('L','U'):entry['by_role'][role]['equal']=False
 finally:
  for stream in streams:stream.close()
 return result

def parameter_motion(source,initial,actual,torch):
 final=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if (final['epoch']!=20 or final['a1_ema_successful_updates']!=4440
     or final['args']['baseline_ckpt'] or final['args']['teacher_ckpt']
     or final['checkpoint_selection']!='final_only' or final.get('ema_model') is None):
  raise ValueError('Own scratch FINAL20/default-head/EMA payload differs')
 state=final['model'];head='id_backbone.encoder.core.id_backbone.cls_head.weight'
 if set(state)!=set(initial) or tuple(state[head].shape)!=(6,160):
  raise ValueError('Native default classifier/state layout differs')
 changed=actual['changed_parameters'];verified={}
 for name,witness in changed.items():
  delta=float((state[name].float()-initial[name].float()).norm())
  if delta<=0 or not math.isfinite(delta) or not math.isclose(delta,float(witness['l2']),rel_tol=1e-6,abs_tol=1e-10):
   raise ValueError('Saved backbone-motion witness differs '+name)
  verified[name]=dict(l2=delta,initial_l2=float(initial[name].float().norm()))
 groups=dict(default_head=[head] if head in verified else [],
             Sinc=[k for k in verified if '.sinc.' in k],
             SIG_TIME=[k for k in verified if '.sequence_pool.' in k],
             other_identity_backbone=[k for k in verified if '.core.id_backbone.' in k
                                     and '.cls_head.' not in k and '.sinc.' not in k])
 if any(not names for names in groups.values()):raise ValueError('Required backbone/default-head updates absent')
 return dict(verified_changed_parameters=verified,groups={k:dict(count=len(v),parameters=v) for k,v in groups.items()},
             joint160_shape=actual['joint160_shape'],checkpoint_epoch=20,optimizer_updates=4440,
             parameter_count=actual['trainable_parameters'],model_forwards=0)

def audit_new(arm,path,root,initial,torch):
 source=path/'source';rr=records(source/'mechanism_steps.jsonl.gz')
 native=records(source/'step_metrics.jsonl.gz');unit=records(source/'unit_steps.jsonl.gz')
 shadow=records(source/'ordinary_draw_steps.jsonl.gz');epochs=records(source/'metrics_epoch.jsonl')
 companion=records(source/'epoch_metrics.jsonl')
 if any(len(v)!=4440 for v in (rr,native,unit,shadow)) or len(epochs)!=20 or len(companion)!=20:
  raise ValueError('Incomplete full4440-step/four-stream/20-epoch records')
 csv_counts={}
 for filename in ('metrics_epoch.csv','epoch_metrics.csv'):
  with (source/filename).open(encoding='utf-8-sig',newline='') as f:csv_counts[filename]=sum(1 for _ in csv.DictReader(f))
 if any(v!=20 for v in csv_counts.values()):raise ValueError('Native/companion epoch CSV incomplete')
 maximum={k:0. for k in ('CE_formula','native_raw','native_extra','native_L','native_first_CE',
                         'labeled_phase_native_total','Sinc_lr','Sinc_decay')}
 expected_as=128 if arm=='RAW_AS' else 0
 raw_first=0;extra_single=0;discarded_sum=0.;discarded_changed=0
 for index,(r,n,u,s) in enumerate(zip(rr,native,unit,shadow),1):
  epoch=(index-1)//222+1
  if any(v['step']!=index for v in (r,n,u,s)) or r['epoch']!=epoch or n['epoch']!=epoch:
   raise ValueError('Ordered continuous training log differs')
  if (r['L_forwards']!=2 or r['As_packets']!=expected_as or r['teacher_packets']!=0
      or r['first_input']!='raw' or r['extra_input']!=('complete practical(raw)' if expected_as else 'raw')
      or r['raw_first_forward_verified'] is not True or r['encoder_raw_MSE']!=0 or r['hook_raw_MSE']!=0
      or s['count']!=128 or s['full_native_draws']!=1 or s['returned_raw_MSE']!=0
      or s['ordinary_output_used_by_student'] is not False):
   raise ValueError('Actual raw/As true-label/ordinary-shadow scope differs')
  raw=scalar(r,'raw_CE');extra=scalar(r,'extra_CE');total=scalar(r,'total_L');m=n['metrics']
  for key,left,right in (
   ('CE_formula',.5*(raw+extra),total),('native_raw',scalar(m,'rawAs/raw_CE'),raw),
   ('native_extra',scalar(m,'rawAs/extra_CE'),extra),('native_L',scalar(m,'rawAs/total_L'),total),
   ('native_first_CE',scalar(m,'train/loss_tx_labeled'),raw)):
   maximum[key]=max(maximum[key],abs(left-right))
  if epoch<=13:maximum['labeled_phase_native_total']=max(maximum['labeled_phase_native_total'],abs(scalar(m,'train/loss')-total))
  if scalar(m,'rawAs/raw_first_forward_verified')!=1 or scalar(m,'train/a1_ema_successful_updates')!=index:
   raise ValueError('Native first-raw or ownEMA update sequence differs')
  if epoch>13 and ('train/pseudo_truth_available' not in m or scalar(m,'train/pseudo_truth_available')!=0):
   raise ValueError('Hidden U truth boundary differs')
  if any(float(m.get(k,0) or 0)!=0 for k in ('train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')):
   raise ValueError('Native training skipped nonfinite updates')
  for key in ('train/loss','train/grad_before_clip','train/grad_total'):scalar(m,key)
  expected_lr=1e-6+(2e-4-1e-6)*(1+math.cos(math.pi*(epoch-1)/19))/2
  if u['physical_clip_calls']!=index or u['sinc_weight_decay']!=0:
   raise ValueError('Physical Sinc clip/centered optimizer step differs')
  maximum['Sinc_lr']=max(maximum['Sinc_lr'],abs(scalar(u,'base_lr')-expected_lr),abs(scalar(u,'sinc_lr')-expected_lr*1000))
  maximum['Sinc_decay']=max(maximum['Sinc_decay'],abs(scalar(u,'physical_decay_alpha_lambda')-expected_lr*1e-4))
  raw_first+=1;extra_single+=int(r['L_forwards']==2)
  discarded_sum+=scalar(s,'discardedordinary_MSE');discarded_changed+=int(s['discardedordinary_changed_packets'])
 if max(maximum.values())>2e-5:raise ValueError('Full formula/native consumption replay mismatch '+str(maximum))
 actual=read(source/'training_result.json');completion=read(source/'completion.json')
 if (completion['epoch']!=20 or completion['optimizer_steps']!=4440 or completion['head_fit_updates']!=0
     or completion['target_access'] or completion['final_source_metrics']['source_val_count']!=27000
     or actual['L_packets']!=1136640 or actual['student_train_packets']!=1533540
     or actual['teacher_packets']!=0 or actual['own_EMA_train_packets']!=0
     or actual['own_EMA_eval_packets']!=396900 or actual['ordinary_shadow_packets']!=568320
     or actual['raw_first_forward_verified_steps']!=4440 or actual['U_packets']!=396900
     or actual['CROSS_RX'] or actual['RDrop'] or actual['external_teacher'] or actual['KD']):
  raise ValueError('Actual complete raw/full-source/default-head/U/EMA budget differs')
 if (0 if actual['channel'] is None else actual['channel']['packets'])!=expected_as*4440:
  raise ValueError('Full As exposure budget differs')
 curves=[]
 for ep in range(20):
  rows=rr[ep*222:(ep+1)*222];steps=native[ep*222:(ep+1)*222]
  curves.append(dict(epoch=ep+1,raw_CE=float(np.mean([r['raw_CE'] for r in rows])),
      extra_CE=float(np.mean([r['extra_CE'] for r in rows])),total_L=float(np.mean([r['total_L'] for r in rows])),
      native_total_loss=float(np.mean([r['metrics']['train/loss'] for r in steps])),
      grad_before_clip=float(np.mean([r['metrics']['train/grad_before_clip'] for r in steps])),
      U_selected_mean=None if ep<13 else float(np.mean([r['metrics'].get('train/pseudo_selected',0) or 0 for r in steps])),
      base_lr=unit[ep*222]['base_lr']))
 text=(root/'logs'/(arm+'-s2026092701.log')).read_text(encoding='utf8',errors='replace')
 error_lines=[s for s in text.splitlines() if re.search(r'Traceback|CUDA out of memory|FloatingPointError|^Killed$|RuntimeError|ModuleNotFoundError',s)]
 motion=parameter_motion(source,initial,actual,torch)
 return dict(complete_records=dict(mechanism=len(rr),native=len(native),unit=len(unit),ordinary_shadow=len(shadow),
      native_epochs=len(epochs),companion_epochs=len(companion),csv_rows=csv_counts),
      max_formula_errors=maximum,all4440_steps_checked=True,hidden_U_steps=1554,
      ownEMA_all4440_updates_exact=True,raw_first_verified_steps=raw_first,
      extra_single_forward_steps_from_hook_budget=extra_single,
      extra_forward_evidence='Each real hook asserted extra_seen==1; saved L_forwards2 and total counters agree; no new encoder observation',
      ordinary_shadow=dict(calls=4440,packets=568320,mean_discarded_MSE=discarded_sum/4440,
          changed_packet_draws=discarded_changed,returned_raw_MSE_max=0.,NN=0),
      sequence=sequence_counts(source/'training_sequence.jsonl.gz'),motion=motion,
      costs=dict(L_student_NN=1136640,U_student_NN=396900,ownEMA_NN=396900,external_teacher_NN=0,
          source_training_eval_NN=actual['student_eval_packets'],As_observations=expected_as*4440,
          source_qualification_NN=12600,formal_prediction_NN=1176002,
          training_wall_seconds=completion['elapsed_seconds']),
      curves=curves,native_epochs=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in epochs],
      native_log_lines=len(text.splitlines()),technical_error_line_count=len(error_lines),technical_error_lines=error_lines,
      ordinary=read(source/'ordinary_augmentation_actual.json'),channel=actual['channel'])

def collect(c):
 root=Path(c['project'])/'runs'/c['run_id'];ref=Path(c['project'])/'runs'/c['reference_run']
 terminal=read(root/'complete.json');decision=read(root/'source_matrix_decision.json')
 if terminal['status']!='ANALYZED' or decision['formal_scores_read'] is not False:
  raise ValueError('Complete matrix and frozen source-first decision required')
 paths={a:(ref if a in ('B0','B1') else root)/(a+'-s2026092701') for a in ('B0','B1','RAW_NOAS','RAW_AS')}
 # Check every prediction/scoring terminal before loading target decisions/truth.
 for arm,p in paths.items():
  prediction=read(p/'prediction/complete.json');score=read(p/'scoring/complete.json')
  if (prediction['status']!='PREDICTIONS_COMPLETE' or prediction['count']!=168000
      or prediction['views']!=c['views'] or prediction['truth_read'] or prediction['query_fit']
      or score['status']!='SCORED_COMPLETE' or not score['truth_last']):
   raise ValueError('Fixed all7 prediction and independent scoring required')
  if arm.startswith('RAW_') and decision['timestamp']>=(p/'scoring/complete.json').stat().st_mtime:
   raise ValueError('Source decision did not precede new scoring')
 runner=Path(c['project'])/'releases'/c['release']/'experiments/cvs_F_rawAs20/runner.py'
 spec=importlib.util.spec_from_file_location('_rawAs_readonly_analysis',runner)
 boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.bootstrap()
 import torch
 torch.set_num_threads(2)
 initials={a:torch.load(p/'source/initial_smoke.pt',map_location='cpu',weights_only=False)['model'] for a,p in paths.items()}
 equal=all(set(v)==set(initials['B0']) and all(torch.equal(t,v[k]) for k,t in initials['B0'].items()) for v in initials.values())
 if not equal:raise ValueError('Four same-seed pure-F scratch initial states differ')
 result=dict(status='ANALYZED',run_id=c['run_id'],model_seed=2026092701,model_seed_count=1,sample_SD=None,
      source_decision=decision,initial_all_four_exact_equal=equal,
      initial_state_entries=len(initials['B0']),logs={},source={},source_pairs={},formal={},paired={},
      extra_model_NN=0,new_predictions=0,new_scoring_jobs=0,target_feedback=False,goal_complete=False)
 for arm in ('RAW_NOAS','RAW_AS'):result['logs'][arm]=audit_new(arm,paths[arm],root,initials[arm],torch)
 result['training_sequence_pairs']=sequence_pairs(paths)
 result['training_sequences_all_four_equal']=all(v['equal'] for v in result['training_sequence_pairs'].values())
 channels={a:read(paths[a]/'source/training_result.json')['channel'] for a in ('B1','RAW_AS')}
 result['B1_RAW_AS_physical_training_summary_equal']=channels['B1']==channels['RAW_AS']
 result['reference_costs']={}
 for arm in ('B0','B1'):
  actual=read(paths[arm]/'source/training_result.json');done=read(paths[arm]/'source/completion.json')
  result['reference_costs'][arm]=dict(L_student_NN=actual['L_packets'],
    student_train_NN=actual['student_train_packets'],student_eval_NN=actual['student_eval_packets'],
    external_teacher_shadow_NN=actual['teacher_packets'],
    source_qualification_NN=12600,formal_prediction_NN=1176002,
    training_wall_seconds=done['elapsed_seconds'],reused=True)
 source_data={};source_ids=None;source_y=None
 for arm,p in paths.items():
  with np.load(p/'source_qualification/predictions.npz',allow_pickle=False) as f:
   ids=f['ids'].copy();y=f['TX'].copy();clean=f['clean'].copy();sat=f['satellite'].copy()
   if ids.shape!=(1800,) or y.shape!=(1800,) or clean.shape!=(1800,6) or sat.shape!=(2,3,1800,6) or not np.isfinite(sat).all() or not np.isfinite(clean).all():
    raise ValueError('Complete fixed source default-head observations required')
   if source_ids is None:source_ids=ids;source_y=y
   elif not np.array_equal(ids,source_ids) or not np.array_equal(y,source_y):raise ValueError('Source legal labels/physical pairs differ')
  done=read(p/'source/completion.json');rates=(sat.argmax(-1)==y).mean(-1)
  if done['final_source_metrics']['source_val_count']!=27000 or not np.allclose(rates,decision['readouts'][arm]['source_accuracies'],atol=1e-12,rtol=0):
   raise ValueError('Frozen source decision readout differs')
  source_data[arm]=dict(clean=clean.argmax(1),satellite=sat.argmax(-1))
  result['source'][arm]=dict(fullVclean=done['final_source_metrics']['source_val_accuracy'],
      fullVworstRX=done['final_source_metrics']['source_val_worst_rx'],panel_clean=confusion_metrics(y,clean.argmax(1)),
      source_accuracies=rates.tolist(),midlow=float(rates[:,1:].mean()))
 gain=100*(result['source']['RAW_AS']['midlow']-result['source']['B1']['midlow'])
 net={a:100*(result['source']['RAW_AS']['fullVclean']-result['source'][a]['fullVclean']) for a in ('B0','RAW_NOAS')}
 if abs(gain-decision['midlow_increment_pp'])>1e-9 or any(abs(net[a]-decision['fullVclean_net_pp'][a])>1e-9 for a in net) or bool(gain>=.5 and min(net.values())>=-.2)!=decision['source_qualified']:
  raise ValueError('Predeclared source decision recount differs')
 result['source_criterion_independently_recomputed']=True
 for baseline in ('B0','B1','RAW_NOAS'):
  result['source_pairs'][baseline]=dict(clean=paired(source_y,source_data[baseline]['clean'],source_data['RAW_AS']['clean']),
    satellite={name:paired(np.tile(source_y,2),source_data[baseline]['satellite'][:,j].reshape(-1),source_data['RAW_AS']['satellite'][:,j].reshape(-1))
               for j,name in enumerate(('practical_low_suburban','practical_mid_urban','practical_low_urban'))})
 # Only terminal frozen predictions enter the following truth-side read-only recount.
 predictions={};ids=None
 for arm,p in paths.items():
  with np.load(p/'prediction/predictions.npz',allow_pickle=False) as f:
   current=f['ids'].copy()
   if ids is None:ids=current
   elif not np.array_equal(current,ids):raise ValueError('Formal physical pairs differ')
   predictions[arm]={v:f[v].copy() for v in c['views']}
   if any(a.shape!=(168000,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>5 for a in predictions[arm].values()):
    raise ValueError('Fixed formal class decisions differ')
 from experiments.cvs_early_geometry.evaluate import truth_arrays
 truth=read(Path(c['project'])/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')
 y,rx,days=truth_arrays(truth,ids)
 for arm,p in paths.items():
  saved=read(p/'scoring/scores.json');overall={r['view']:r for r in saved['results'] if r['dimension']=='overall'}
  if not saved['truth_last'] or set(overall)!=set(c['views']):raise ValueError('Existing seven-view independent scores incomplete')
  metrics={};strata={}
  for view in c['views']:
   pred=predictions[arm][view];metric=confusion_metrics(y,pred)
   if any(abs(metric[k]-overall[view][k])>1e-12 for k in ('accuracy','macro_f1')):raise ValueError('Saved formal score differs from frozen decisions')
   metrics[view]=metric;strata[view]={}
   for dim,values in (('receiver',rx),('day',days),('transmitter',y)):
    strata[view][dim]={str(v):confusion_metrics(y[values==v],pred[values==v]) for v in sorted(set(values))}
   metric['worst_RX']=min(r['accuracy'] for r in strata[view]['receiver'].values())
  result['formal'][arm]=dict(overall=metrics,strata=strata,reused=arm in ('B0','B1'))
 for arm,baseline in (('RAW_AS','B0'),('RAW_AS','B1'),('RAW_AS','RAW_NOAS'),('RAW_NOAS','B0')):
  result['paired'][arm+'_minus_'+baseline]={v:paired(y,predictions[baseline][v],predictions[arm][v]) for v in c['views']}
 result['suburban_gap_pp']={a:{v:100*(result['formal'][a]['overall']['clean']['accuracy']-result['formal'][a]['overall'][v]['accuracy']) for v in c['views'][1:4]} for a in paths}
 b0clean=predictions['B0']['clean']==y;n=int(b0clean.sum())
 result['fixed_B0_clean_correct_cohort']=dict(n=n,definition='One fixed set where B0 clean is correct; no sample selection or model updates',
   rows={a:{v:dict(retained_correct=int((predictions[a][v][b0clean]==y[b0clean]).sum()),
                 retention=float((predictions[a][v][b0clean]==y[b0clean]).mean())) for v in c['views']} for a in paths})
 result['budget_comparability']=dict(epochs_each=20,updates_each=4440,data='L6300/U56700/V27000',
   B0_L_student_NN=568320,B1_L_student_NN=1136640,new_row_L_student_NN=1136640,
   B1_external_teacher_shadow_NN=568320,new_rows_external_teacher_NN=0,
   new_rows_ordinary_shadow_packets_each=568320,extra_analysis_NN=0,
   note='Shadow augmentation keeps its own computation; no wall-time equality or whole-training RNG equality assumed')
 return result

print(json.dumps(collect(CONFIG),allow_nan=False))
'''


def configuration(project):
    return dict(project=project, run_id=RUN, reference_run=REFERENCE_RUN,
                release=RELEASE, views=list(VIEWS))


def remote_helper(runtime):
    """Load the existing verified standalone SSH/JSON reader, never a new transport."""
    explicit = runtime.get('analysis_remote_helper')
    candidates = ([Path(explicit)] if explicit else []) + [
        ROOT/'experiments/cvs_sameiq_experts/publish.py',
        ROOT.parent/'phase1_sameiq_experts_20261019/experiments/cvs_sameiq_experts/publish.py',
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError('Existing cvs_sameiq_experts.publish.remote helper unavailable')
    spec = importlib.util.spec_from_file_location('_existing_rawAs_analysis_remote', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.remote


def main(runtime, output, render_figures=False):
    rt = json.loads(Path(runtime).read_text(encoding='utf-8-sig'))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    result = remote_helper(rt)(SCRIPT.replace('CONFIG', repr(configuration(rt['project']))), rt)
    (out/'complete_analysis.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    if render_figures:
        render(result, out)
    print(json.dumps(dict(status=result['status'], complete_training_checked=True,
                          initial_all_four_exact_equal=result['initial_all_four_exact_equal'],
                          training_sequences_all_four_equal=result['training_sequences_all_four_equal'],
                          extra_model_NN=0, source_decision=result['source_decision']['decision'])))
    return result


def render(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    stems = ('seven_views_and_full_training', 'paired_errors_and_clean_retention')
    if any((out/(stem+suffix)).exists() for stem in stems for suffix in ('.png', '.pdf')):
        raise FileExistsError('Existing analysis figures are preserved; choose a new output directory')
    colors = dict(B0='#929292', B1='#356d9b', RAW_NOAS='#b77d31', RAW_AS='#19856b')
    labels = dict(B0='Original NoAs', B1='Ordinary + As', RAW_NOAS='Raw + raw', RAW_AS='Raw + As')
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), layout='constrained')
    for arm in colors:
        values = result['formal'][arm]['overall']
        axes[0, 0].plot(range(7), [100*values[v]['accuracy'] for v in VIEWS], '-o',
                        label=labels[arm], color=colors[arm], markersize=4)
    for arm in ('RAW_NOAS', 'RAW_AS'):
        curve = result['logs'][arm]['curves']
        epoch = [v['epoch'] for v in curve]
        for ax, key in ((axes[0, 1], 'raw_CE'), (axes[1, 0], 'extra_CE'), (axes[1, 1], 'total_L')):
            ax.plot(epoch, [v[key] for v in curve], label=labels[arm], color=colors[arm])
    axes[0, 0].set(xticks=range(7), xticklabels=['clean', 'high', 'mid', 'low sub', 'high urb', 'mid urb', 'low urb'], ylabel='Accuracy (%)')
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 1].set(xlabel='Epoch', ylabel='First-slot raw CE')
    axes[1, 0].set(xlabel='Epoch', ylabel='Extra-slot raw / As CE')
    axes[1, 1].set(xlabel='Epoch', ylabel='0.5 raw CE + 0.5 extra CE')
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.axvline(13.5, color='#777777', linestyle=':', linewidth=.8)
        ax.legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Pure F scratch E20; one model seed; fixed seven views')
    for suffix in ('.png', '.pdf'):
        fig.savefig(out/(stems[0]+suffix), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    xs = list(range(7))
    for index, baseline in enumerate(('B0', 'B1', 'RAW_NOAS')):
        values = result['paired']['RAW_AS_minus_'+baseline]
        axes[0].plot(xs, [values[v]['net_pp'] for v in VIEWS], '-o', markersize=4,
                     label='Raw + As minus '+labels[baseline], color=colors[baseline])
    axes[0].axhline(0, color='#777777', linewidth=.8)
    cohort = result['fixed_B0_clean_correct_cohort']
    for arm in colors:
        axes[1].plot(xs, [100*cohort['rows'][arm][v]['retention'] for v in VIEWS], '-o',
                     label=labels[arm], color=colors[arm], markersize=4)
    for ax in axes:
        ax.set(xticks=xs, xticklabels=['clean', 'high', 'mid', 'low sub', 'high urb', 'mid urb', 'low urb'])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('Paired accuracy difference (percentage points)')
    axes[1].set_ylabel('Correct retained on fixed B0-clean-correct set (%)')
    fig.suptitle('Frozen prediction recount; model-seed n=1')
    for suffix in ('.png', '.pdf'):
        fig.savefig(out/(stems[1]+suffix), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime')
    parser.add_argument('--output', required=True)
    parser.add_argument('--render-cache')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if args.render_cache:
        render(json.loads(Path(args.render_cache).read_text(encoding='utf-8-sig')), args.output)
    else:
        if not args.runtime:
            parser.error('--runtime is required for collection')
        main(args.runtime, args.output, args.render)
