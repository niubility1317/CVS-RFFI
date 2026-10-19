"""Complete two-row risk replay, four-reference pairing and fixed seven-view report."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.cvs_sameiq_experts.publish import remote
RUN='20261019-phase1-F-sceneRisk-e20-s2026092701-m2-r01'
SCRIPT=r'''
import gzip,json,itertools,math,re,importlib.util
from pathlib import Path
import numpy as np
c=CONFIG;root=Path(c['project'])/'runs'/c['run_id'];ref=Path(c['project'])/'runs'/c['reference_run'];views=c['views']
def read(p):return json.loads(p.read_text())
def records(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else p.open() as f:return [json.loads(s) for s in f if s.strip()]
if not (root/'complete.json').is_file():raise ValueError('Registered training/prediction/scoring incomplete')
decision=read(root/'source_matrix_decision.json')
if decision['formal_scores_read'] is not False:raise ValueError('Source decision depends on scores')
runner=Path(c['project'])/'releases/cvs_F_sceneRisk_E20_20261019_r01/experiments/cvs_scene_risk20/runner.py';spec=importlib.util.spec_from_file_location('_scene_risk_readonly_analysis',runner);boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.bootstrap()
import torch
torch.set_num_threads(2)
paths={a:(ref if a in ('B0','B1') else root)/(a+'-s2026092701') for a in ('B0','B1','BALANCED','RISK')}
initials={a:torch.load(p/'source/initial_smoke.pt',map_location='cpu',weights_only=False)['model'] for a,p in paths.items()}
same=all(set(q)==set(initials['B0']) and all(torch.equal(v,q[k]) for k,v in initials['B0'].items()) for q in initials.values())
if not same:raise ValueError('Matched scratch F initial states differ')
result=dict(summary=read(root/'analysis.json'),initial_all_four_exact_equal=same,logs={},paired={},source_pairs={})
for arm in ('BALANCED','RISK'):
 p=paths[arm];source=p/'source';rr=records(source/'mechanism_steps.jsonl.gz');native=records(source/'step_metrics.jsonl.gz');unit=records(source/'unit_steps.jsonl.gz');ep=records(source/'metrics_epoch.jsonl')
 if len(rr)!=4440 or len(native)!=4440 or len(unit)!=4440 or len(ep)!=20:raise ValueError('Incomplete full training log')
 errors=[];maxerr=dict(group_mean=0.,logq=0.,weights=0.,loss=0.,native=0.);last=np.zeros(3);curves=[];countsall=np.zeros(3,dtype=int);global_min=1.;global_max=0.
 for index,(r,n) in enumerate(zip(rr,native),1):
  if r['step']!=index or n['step']!=index or r['epoch']!=(index-1)//222+1 or r['teacher_packets']!=0 or r['As_packets']!=128 or r['L_forwards']!=2:raise ValueError('Actual ordered per-step budget differs')
  ce=np.asarray(r['packet_CE'],float);g=np.asarray(r['scene_index'],int);counts=np.bincount(g,minlength=3);present=counts>0;means=np.asarray([ce[g==i].mean() if present[i] else 0. for i in range(3)])
  if ce.shape!=(128,) or g.shape!=(128,) or not np.isfinite(ce).all() or (ce<0).any() or not np.array_equal(counts,r['counts']):raise ValueError('Every legal As packet must remain')
  q=last+.01*means;q=q-(q.max()+np.log(np.exp(q-q.max()).sum()));s=np.exp(q-q.max())*present;s=s/s.sum();u=present.astype(float)/present.sum();w=u if arm=='BALANCED' else .5*(u+s)
  expected=float((w*means).sum());total=.5*(r['ordinary_CE']+expected);applied=np.asarray(r['applied_weights'])
  maxerr['group_mean']=max(maxerr['group_mean'],float(np.max(np.abs(means-r['means']))));maxerr['logq']=max(maxerr['logq'],float(np.max(np.abs(last-r['logq_before']))),float(np.max(np.abs(q-r['logq_after']))));maxerr['weights']=max(maxerr['weights'],float(np.max(np.abs(w-applied))));maxerr['loss']=max(maxerr['loss'],abs(expected-r['weighted_CE']),abs(total-r['total_L']))
  # Native telemetry is the value actually inserted into the trained loss.
  m=n['metrics'];maxerr['native']=max(maxerr['native'],abs(m['scene_risk/ordinary_CE']-r['ordinary_CE']),abs(m['scene_risk/As_CE']-r['weighted_CE']),abs(m['scene_risk/total_L']-r['total_L']))
  if (applied[present]<=0).any() or np.any(applied[~present]!=0) or abs(applied.sum()-1)>1e-6:raise ValueError('A legal labeled As lost positive supervision')
  if len(r['TX'])!=128 or min(r['TX'])<0 or max(r['TX'])>5:raise ValueError('Legal labels differ')
  countsall+=counts;global_min=min(global_min,float(applied[present].min()));global_max=max(global_max,float(applied.max()));last=np.asarray(r['logq_after'])
 if max(maxerr.values())>2e-5:raise ValueError('Full independent risk replay mismatch: '+str(maxerr))
 if not np.allclose(last,read(source/'scene_risk_state.json')['logq'],atol=1e-12):raise ValueError('Final risk continuation state differs')
 for e in range(20):
  rs=rr[e*222:(e+1)*222];curves.append(dict(epoch=e+1,ordinary_CE=float(np.mean([r['ordinary_CE'] for r in rs])),packet_As_CE=float(np.mean([r['extra_CE'] for r in rs])),weighted_As_CE=float(np.mean([r['weighted_CE'] for r in rs])),total_L=float(np.mean([r['total_L'] for r in rs])),weights=np.mean([r['applied_weights'] for r in rs],axis=0).tolist(),scene_CE=np.mean([r['means'] for r in rs],axis=0).tolist(),scene_counts=np.sum([r['counts'] for r in rs],axis=0).tolist()))
 usteps=[r for r in native if r['step']>13*222]
 if len(usteps)!=1554 or any(float(r['metrics'].get('train/pseudo_truth_available',-1) or 0)!=0 for r in usteps) or any(float(r['metrics'].get('train/a1_ema_successful_updates',-1))!=r['step'] for r in native):raise ValueError('Hidden U/original ownEMA differs')
 seqcounts={'L':0,'U':0};seqpackets={'L':0,'U':0}
 for r in records(source/'training_sequence.jsonl.gz'):seqcounts[r['role']]+=1;seqpackets[r['role']]+=len(r['ids'])
 log=(root/'logs'/(arm+'-s2026092701.log')).read_text();errors=[s for s in log.splitlines() if re.search(r'Traceback|CUDA out of memory|FloatingPointError|^Killed$|RuntimeError|ModuleNotFoundError',s)]
 result['logs'][arm]=dict(complete_records=dict(mechanism=len(rr),native=len(native),unit=len(unit),epochs=len(ep)),all_risk_steps_independently_replayed=True,max_formula_errors=maxerr,all_As_positive_CE=True,actual_scene_counts=countsall.tolist(),min_applied_group_weight=global_min,max_applied_group_weight=global_max,final_logq=last.tolist(),curves=curves,native_epochs=ep,hidden_U_steps=len(usteps),ownEMA_all_updates_exact=True,sequence_batches=seqcounts,sequence_packets=seqpackets,errors=errors,actual=read(source/'training_result.json'),completion=read(source/'completion.json'),ordinary=read(source/'ordinary_augmentation_actual.json'),prediction_completion=read(p/'prediction/complete.json'))
 if errors:raise ValueError('Training logs include a technical failure')
result['training_sequences_all_four_equal']=True
streams=[gzip.open(paths[a]/'source/training_sequence.jsonl.gz','rt') for a in paths]
try:
 for items in itertools.zip_longest(*streams):
  if any(v!=items[0] for v in items[1:]):result['training_sequences_all_four_equal']=False;break
finally:
 for f in streams:f.close()
channels={a:read(p/'source/training_result.json')['channel'] for a,p in paths.items() if a!='B0'}
result['physical_training_summary_equal_B1_BALANCED_RISK']=all(v==channels['B1'] for v in channels.values())
def pair(y,b,m):return dict(count=len(y),corrected=int(((m==y)&(b!=y)).sum()),damaged=int(((m!=y)&(b==y)).sum()),both_correct=int(((m==y)&(b==y)).sum()),both_wrong=int(((m!=y)&(b!=y)).sum()),net_pp=float(100*((m==y).mean()-(b==y).mean())))
from experiments.cvs_early_geometry.evaluate import truth_arrays
with np.load(paths['RISK']/'prediction/predictions.npz',allow_pickle=False) as f:ids=f['ids'].copy();mp={v:f[v].copy() for v in views}
y,rx,day=truth_arrays(read(Path(c['project'])/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'),ids)
for a,p in paths.items():
 with np.load(p/'prediction/predictions.npz',allow_pickle=False) as f:
  if not np.array_equal(f['ids'],ids):raise ValueError('Formal paired physical IDs differ')
  result['paired'][a]={v:pair(y,f[v],mp[v]) for v in views}
 if a in ('BALANCED','RISK') and decision['timestamp']>= (p/'scoring/complete.json').stat().st_mtime:raise ValueError('Source decision did not precede new scoring')
with np.load(paths['RISK']/'source_qualification/predictions.npz',allow_pickle=False) as f:sy=f['TX'].copy();sid=f['ids'].copy();mc=f['clean'].argmax(1);ms=f['satellite'].argmax(-1)
for a,p in paths.items():
 with np.load(p/'source_qualification/predictions.npz',allow_pickle=False) as f:
  if not np.array_equal(f['ids'],sid):raise ValueError('Source paired IDs differ')
  result['source_pairs'][a]=dict(clean=pair(sy,f['clean'].argmax(1),mc),urban={str(j):pair(np.tile(sy,2),f['satellite'][:,j].argmax(-1).reshape(-1),ms[:,j].reshape(-1)) for j in (1,2)})
print(json.dumps(result))
'''
def main(runtime,output):
 rt=json.loads(Path(runtime).read_text(encoding='utf8'));out=Path(output);out.mkdir(parents=True,exist_ok=False)
 c=dict(project=rt['project'],run_id=RUN,reference_run='20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01',views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'])
 result=remote(SCRIPT.replace('CONFIG',repr(c)),rt);(out/'complete_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');render(result,out)
 print(json.dumps(dict(status=result['summary']['status'],source=result['summary']['source_decision']['decision'],differences=result['summary']['differences'],initial_equal=result['initial_all_four_exact_equal'],training_sequences_equal=result['training_sequences_all_four_equal'],physical_As_equal=result['physical_training_summary_equal_B1_BALANCED_RISK'])))
def render(result,out):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors={'B0':'#888888','B1':'#467bad','BALANCED':'#bf8d35','RISK':'#178a73'};fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');views=list(result['summary']['rows']['RISK']['overall'])
 for a,color in colors.items():axes[0,0].plot(range(7),[100*result['summary']['rows'][a]['overall'][v]['accuracy'] for v in views],'-o',label=a,color=color,ms=4)
 for a in ('BALANCED','RISK'):
  curve=result['logs'][a]['curves'];x=[r['epoch'] for r in curve];axes[0,1].plot(x,[r['ordinary_CE'] for r in curve],label=a,color=colors[a]);axes[1,0].plot(x,[r['packet_As_CE'] for r in curve],label=a,color=colors[a])
 for j,name in enumerate(('low suburban','mid urban','low urban')):axes[1,1].plot(range(1,21),[r['weights'][j] for r in result['logs']['RISK']['curves']],label=name)
 axes[0,0].set(xticks=range(7),xticklabels=['clean','high','mid','low sub','high urb','mid urb','low urb'],ylabel='Accuracy (%)');axes[0,0].legend(frameon=False);axes[0,1].set(xlabel='Epoch',ylabel='Ordinary CE');axes[1,0].set(xlabel='Epoch',ylabel='Unweighted packet As CE');axes[1,1].set(xlabel='Epoch',ylabel='RISK mean applied scene weight',ylim=(0,0.7));axes[1,1].legend(frameon=False)
 for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
 fig.savefig(out/'seven_views_and_scene_risk.png',dpi=180);fig.savefig(out/'seven_views_and_scene_risk.pdf');plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime');p.add_argument('--output',required=True);p.add_argument('--render-cache');a=p.parse_args()
 if a.render_cache:render(json.loads(Path(a.render_cache).read_text(encoding='utf8')),Path(a.output))
 else:main(a.runtime,a.output)
