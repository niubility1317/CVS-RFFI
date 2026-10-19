"""Complete conditional-risk replay and independently recounted seven-view analysis."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.cvs_sameiq_experts.publish import remote
RUN='20261019-phase1-F-conditionalTail-e20-s2026092701-m2-r01'
SCRIPT=r'''
import gzip,json,itertools,re,importlib.util
from pathlib import Path
import numpy as np
c=CONFIG;root=Path(c['project'])/'runs'/c['run_id'];ref=Path(c['project'])/'runs'/c['reference_run'];views=c['views']
def read(p):return json.loads(p.read_text())
def records(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else p.open() as f:return [json.loads(s) for s in f if s.strip()]
if not c['source_only'] and not (root/'complete.json').is_file():raise ValueError('Registered training/prediction/scoring incomplete')
decision=read(root/'source_matrix_decision.json')
if decision['formal_scores_read'] is not False:raise ValueError('Source decision depends on scores')
runner=Path(c['project'])/'releases/cvs_F_conditionalTail_E20_20261019_r01/experiments/cvs_conditional_tail20/runner.py';spec=importlib.util.spec_from_file_location('_conditional_tail_readonly_analysis',runner);boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.bootstrap()
import torch
torch.set_num_threads(2)
arms=('B0','B1','CE2','TAIL2');newarms=arms[2:]
paths={a:(ref if a in arms[:2] else root)/(a+'-s2026092701') for a in arms}
initials={a:torch.load(p/'source/initial_smoke.pt',map_location='cpu',weights_only=False)['model'] for a,p in paths.items()}
same=all(set(q)==set(initials['B0']) and all(torch.equal(v,q[k]) for k,v in initials['B0'].items()) for q in initials.values())
if not same:raise ValueError('Matched scratch F initial states differ')
summary=dict(status='SOURCE_AUDITED',run_id=c['run_id'],source_decision=decision,formal_scores_read=False) if c['source_only'] else read(root/'analysis.json')
result=dict(summary=summary,initial_all_four_exact_equal=same,logs={},paired={},source_pairs={})
for arm in newarms:
 p=paths[arm];source=p/'source';rr=records(source/'mechanism_steps.jsonl.gz');native=records(source/'step_metrics.jsonl.gz');unit=records(source/'unit_steps.jsonl.gz');ep=records(source/'metrics_epoch.jsonl')
 if len(rr)!=4440 or len(native)!=4440 or len(unit)!=4440 or len(ep)!=20:raise ValueError('Incomplete full training log')

 maxerr=dict(weights=0.,risk=0.,total_L=0.,native=0.);curves=[];global_min=float('inf');global_max=0.;hard=[0,0,0]
 for index,(r,n) in enumerate(zip(rr,native),1):
  if r['step']!=index or n['step']!=index or r['epoch']!=(index-1)//222+1 or r['teacher_packets']!=0 or r['As_packets']!=256 or r['L_forwards']!=3 or r['same_physical_pairs']!=128 or r['second_draw_namespace_step']!=4441-index:raise ValueError('Actual ordered two-draw packet/namespace budget differs')
  ce=np.asarray(r['draw_CE'],float);weights=np.asarray(r['draw_weights'],float);tx=np.asarray(r['TX'],int);which=np.asarray(r['harder_draw'],int)
  if ce.shape!=(2,128) or weights.shape!=(2,128) or tx.shape!=(128,) or which.shape!=(128,) or not np.isfinite(ce).all() or (ce<0).any() or set(tx)-set(range(6)):raise ValueError('Every two-draw true-label packet must remain')
  delta=ce[0]-ce[1];winner=np.stack([(delta>0)+.5*(delta==0),(delta<0)+.5*(delta==0)])
  w=np.full_like(ce,.5) if arm=='CE2' else .25+.5*winner
  expected=float(np.mean(np.sum(w*ce,axis=0)));total=.5*(r['ordinary_CE']+expected)
  maxerr['weights']=max(maxerr['weights'],float(np.max(abs(w-weights))),float(np.max(abs(weights.sum(0)-1))))
  maxerr['risk']=max(maxerr['risk'],abs(expected-r['risk_CE']),abs(float(ce.mean())-r['conditional_mean_CE']),abs(float(ce.max(0).mean())-r['conditional_max_CE']),abs(expected-r['extra_CE']))
  maxerr['total_L']=max(maxerr['total_L'],abs(total-r['total_L']))
  m=n['metrics'];maxerr['native']=max(maxerr['native'],abs(m['conditional_tail/ordinary_CE']-r['ordinary_CE']),abs(m['conditional_tail/As_CE']-r['risk_CE']),abs(m['conditional_tail/total_L']-r['total_L']))
  if (weights<=0).any() or not np.array_equal(which,np.sign(delta).astype(int)):raise ValueError('Positive CE or true harder-draw assignment differs')
  for j,value in enumerate((-1,0,1)):hard[j]+=int((which==value).sum())
  global_min=min(global_min,float(weights.min()));global_max=max(global_max,float(weights.max()))
 if any(v>2e-5 for v in maxerr.values()):raise ValueError('Independent two-draw objective replay differs: '+str(maxerr))
 for e in range(20):
  rs=rr[e*222:(e+1)*222];curves.append(dict(epoch=e+1,ordinary_CE=float(np.mean([r['ordinary_CE'] for r in rs])),mean_draw_CE=float(np.mean([r['conditional_mean_CE'] for r in rs])),max_draw_CE=float(np.mean([r['conditional_max_CE'] for r in rs])),risk_CE=float(np.mean([r['risk_CE'] for r in rs])),total_L=float(np.mean([r['total_L'] for r in rs])),first_harder_fraction=float(np.mean([np.mean(np.asarray(r['harder_draw'])>0) for r in rs]))))
 usteps=[r for r in native if r['step']>2886]
 if len(usteps)!=1554 or any(float(r['metrics'].get('train/pseudo_truth_available',-1) or 0)!=0 for r in usteps) or any(float(r['metrics'].get('train/a1_ema_successful_updates',-1))!=r['step'] for r in native):raise ValueError('Hidden U/original ownEMA differs')
 seqcounts={'L':0,'U':0};seqpackets={'L':0,'U':0}
 for r in records(source/'training_sequence.jsonl.gz'):seqcounts[r['role']]+=1;seqpackets[r['role']]+=len(r['ids'])
 log=(root/'logs'/(arm+'-s2026092701.log')).read_text();errors=[s for s in log.splitlines() if re.search(r'Traceback|CUDA out of memory|FloatingPointError|^Killed$|RuntimeError|ModuleNotFoundError',s)]
 actual=read(source/'training_result.json')
 if actual['steps']!=4440 or actual['trainable_parameters']!=900722 or actual['joint160_shape']!=[2,160] or actual['teacher_packets'] or actual['external_teacher_NN'] or actual['L_packets']!=1704960 or actual['As_packets']!=1136640 or actual['second_draw_packets']!=568320 or actual['conditional_pairs']!=568320 or actual['head_fit_updates'] or actual['target_access'] or actual['CROSS_RX'] or actual['RDrop']:raise ValueError('Actual fixed F/defaulthead/three-forward budget differs')
 if actual['channel']['packets']!=568320 or actual['second_channel']['packets']!=568320 or actual['channel']['sessions']!=actual['second_channel']['sessions'] or actual['channel']['dropped'] or actual['second_channel']['dropped']:raise ValueError('Complete propagation/draw/session execution differs')
 result['logs'][arm]=dict(complete_records=dict(mechanism=len(rr),native=len(native),unit=len(unit),epochs=len(ep)),all_risk_steps_independently_replayed=True,max_formula_errors=maxerr,all_As_positive_CE=True,harder_draw_counts=hard,min_draw_weight=global_min,max_draw_weight=global_max,curves=curves,native_epochs=ep,hidden_U_steps=len(usteps),ownEMA_all_updates_exact=True,sequence_batches=seqcounts,sequence_packets=seqpackets,errors=errors,actual=actual,completion=read(source/'completion.json'),ordinary=read(source/'ordinary_augmentation_actual.json'),prediction_completion=None if c['source_only'] else read(p/'prediction/complete.json'))
 if errors:raise ValueError('Training logs include a technical failure')
result['training_sequences_all_four_equal']=True
streams=[gzip.open(paths[a]/'source/training_sequence.jsonl.gz','rt') for a in arms]
try:
 for items in itertools.zip_longest(*streams):
  if any(v!=items[0] for v in items[1:]):result['training_sequences_all_four_equal']=False;break
finally:
 for f in streams:f.close()
channels={a:read(p/'source/training_result.json')['channel'] for a,p in paths.items() if a!='B0'}
result['first_draw_summary_equal_B1_CE2_TAIL2']=all(v==channels['B1'] for v in channels.values())
result['second_draw_summary_equal_new_rows']=result['logs']['CE2']['actual']['second_channel']==result['logs']['TAIL2']['actual']['second_channel']
source_rates={};source_clean={};source_ids=None
for a,p in paths.items():
 with np.load(p/'source_qualification/predictions.npz',allow_pickle=False) as f:
  ids=f['ids'];y=f['TX'];z=f['satellite']
  if ids.shape!=(1800,) or y.shape!=(1800,) or z.shape!=(2,3,1800,6) or not np.isfinite(z).all():raise ValueError('Incomplete source default-head readout')
  if source_ids is None:source_ids=ids.copy()
  elif not np.array_equal(ids,source_ids):raise ValueError('Source physical pair differs')
  source_rates[a]=(z.argmax(-1)==y).mean(-1).tolist()
 done=read(p/'source/completion.json');source_clean[a]=done['final_source_metrics']['source_val_accuracy']
 if done['final_source_metrics']['source_val_count']!=27000:raise ValueError('Full source clean denominator differs')
 if not np.allclose(source_rates[a],decision['readouts'][a]['source_accuracies'],atol=1e-12,rtol=0):raise ValueError('Source decision differs from saved observations')
midlow=lambda a:np.asarray(source_rates[a])[:,1:].mean()
inc={a:100*(midlow('TAIL2')-midlow(a)) for a in ('CE2','B1')};dc=100*(source_clean['TAIL2']-source_clean['B0'])
if any(abs(inc[a]-decision['midlow_increment_pp'][a])>1e-9 for a in inc) or abs(dc-decision['fullVclean_net_pp'])>1e-9 or bool(min(inc.values())>=.5 and dc>=-.2)!=decision['source_qualified']:raise ValueError('Fixed source criterion recount differs')
result['source_criterion_independently_recomputed']=True
if c['source_only']:
 print(json.dumps(result))
 raise SystemExit(0)
def pair(y,b,m):return dict(count=len(y),corrected=int(((m==y)&(b!=y)).sum()),damaged=int(((m!=y)&(b==y)).sum()),both_correct=int(((m==y)&(b==y)).sum()),both_wrong=int(((m!=y)&(b!=y)).sum()),net_pp=float(100*((m==y).mean()-(b==y).mean())))
from experiments.cvs_early_geometry.evaluate import truth_arrays
with np.load(paths['TAIL2']/'prediction/predictions.npz',allow_pickle=False) as f:ids=f['ids'].copy();mp={v:f[v].copy() for v in views}
y,rx,day=truth_arrays(read(Path(c['project'])/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'),ids)
for a,p in paths.items():
 with np.load(p/'prediction/predictions.npz',allow_pickle=False) as f:
  if not np.array_equal(f['ids'],ids):raise ValueError('Formal paired physical IDs differ')
  result['paired'][a]={v:pair(y,f[v],mp[v]) for v in views}
 if a in newarms and decision['timestamp']>= (p/'scoring/complete.json').stat().st_mtime:raise ValueError('Source decision did not precede new scoring')
 # Independently derive every overall/RX/day/TX confusion matrix from fixed decisions.
 score_rows=read(p/'scoring/scores.json')['results'];audited=[]
 with np.load(p/'prediction/predictions.npz',allow_pickle=False) as f:
  for record in score_rows:
   dimension=record['dimension'];value=record['stratum'];pred=f[record['view']]
   mask=np.ones(len(y),bool) if dimension=='overall' else rx==int(value) if dimension=='receiver' else day==int(value) if dimension=='day' else y==int(value) if dimension=='transmitter' else None
   if mask is None or not mask.any():raise ValueError('Unknown or empty registered scoring stratum')
   cm=np.bincount(y[mask]*6+pred[mask],minlength=36).reshape(6,6);acc=float(np.trace(cm)/cm.sum());denom=cm.sum(0)+cm.sum(1);f1=float(np.divide(2*np.diag(cm),denom,out=np.zeros(6),where=denom>0).mean())
   if abs(acc-record['accuracy'])>1e-12 or abs(f1-record['macro_f1'])>1e-12:raise ValueError('Independent full confusion-matrix metrics differ')
   audited.append(dict(view=record['view'],dimension=dimension,stratum=value,count=int(cm.sum()),accuracy=acc,macro_f1=f1,confusion=cm.tolist()))
 result.setdefault('confusion_recount',{})[a]=audited

with np.load(paths['TAIL2']/'source_qualification/predictions.npz',allow_pickle=False) as f:sy=f['TX'].copy();sid=f['ids'].copy();mc=f['clean'].argmax(1);ms=f['satellite'].argmax(-1)
for a,p in paths.items():
 with np.load(p/'source_qualification/predictions.npz',allow_pickle=False) as f:
  if not np.array_equal(f['ids'],sid):raise ValueError('Source paired IDs differ')
  result['source_pairs'][a]=dict(clean=pair(sy,f['clean'].argmax(1),mc),urban={str(j):pair(np.tile(sy,2),f['satellite'][:,j].argmax(-1).reshape(-1),ms[:,j].reshape(-1)) for j in (1,2)})
print(json.dumps(result))
'''

def main(runtime,output,source_only=False):
 rt=json.loads(Path(runtime).read_text(encoding='utf8'));out=Path(output);out.mkdir(parents=True,exist_ok=False)
 c=dict(project=rt['project'],run_id=RUN,source_only=source_only,reference_run='20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01',views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'])
 result=remote(SCRIPT.replace('CONFIG',repr(c)),rt);(out/('source_analysis.json' if source_only else 'complete_analysis.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 if source_only:
  decision=result['summary']['source_decision'];brief={k:decision[k] for k in ('source_qualified','midlow_increment_pp','fullVclean_net_pp','decision')}
  print(json.dumps(dict(status=result['summary']['status'],source=brief,all_training_replayed=True,formal_scores_read=False)));return

 render(result,out)
 print(json.dumps(dict(status=result['summary']['status'],source=result['summary']['source_decision']['decision'],differences=result['summary']['differences'],initial_equal=result['initial_all_four_exact_equal'],training_sequences_equal=result['training_sequences_all_four_equal'],first_draw_equal=result['first_draw_summary_equal_B1_CE2_TAIL2'],second_draw_equal=result['second_draw_summary_equal_new_rows'],all_risk_steps_replayed=True,independent_confusion_records=sum(len(v) for v in result['confusion_recount'].values()))))

def render(result,out):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors={'B0':'#888888','B1':'#467bad','CE2':'#bf8d35','TAIL2':'#178a73'};fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');views=list(result['summary']['rows']['TAIL2']['overall'])
 for a,color in colors.items():axes[0,0].plot(range(7),[100*result['summary']['rows'][a]['overall'][v]['accuracy'] for v in views],'-o',label=a,color=color,ms=4)
 for a in ('CE2','TAIL2'):
  curve=result['logs'][a]['curves'];x=[r['epoch'] for r in curve];axes[0,1].plot(x,[r['ordinary_CE'] for r in curve],label=a,color=colors[a]);axes[1,0].plot(x,[r['mean_draw_CE'] for r in curve],label=a+' mean',color=colors[a]);axes[1,0].plot(x,[r['max_draw_CE'] for r in curve],label=a+' max',color=colors[a],ls='--');axes[1,1].plot(x,[r['first_harder_fraction'] for r in curve],label=a,color=colors[a])
 axes[0,0].set(xticks=range(7),xticklabels=['clean','high','mid','low sub','high urb','mid urb','low urb'],ylabel='Accuracy (%)');axes[0,0].legend(frameon=False);axes[0,1].set(xlabel='Epoch',ylabel='Ordinary CE');axes[1,0].set(xlabel='Epoch',ylabel='Conditional two-draw CE');axes[1,0].legend(frameon=False);axes[1,1].set(xlabel='Epoch',ylabel='First draw harder fraction',ylim=(0,1));axes[1,1].legend(frameon=False)
 for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
 fig.savefig(out/'seven_views_and_conditional_risk.png',dpi=180);fig.savefig(out/'seven_views_and_conditional_risk.pdf');plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime');p.add_argument('--output',required=True);p.add_argument('--render-cache');p.add_argument('--source-only',action='store_true');a=p.parse_args()
 if a.render_cache:render(json.loads(Path(a.render_cache).read_text(encoding='utf8')),Path(a.output))
 else:main(a.runtime,a.output,a.source_only)
