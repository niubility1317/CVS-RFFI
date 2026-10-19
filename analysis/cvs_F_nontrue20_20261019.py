"""Complete-log and paired seven-view readout for one fixed F experiment."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.cvs_sameiq_experts.publish import remote
from experiments.cvs_F_nontrue20 import design as d
SCRIPT=r'''
import gzip,json,math,re,itertools
from pathlib import Path
import numpy as np
c=CONFIG;root=Path(c['project'])/'runs'/c['run_id']
def read(p):return json.loads(p.read_text())
if not (root/'complete.json').is_file():raise ValueError('Immutable training/prediction/scoring matrix incomplete')
if not read(root/'source_matrix_decision.json')['formal_scores_read'] is False:raise ValueError('Source decision order differs')
import importlib.util
runner=Path(c['project'])/'releases/cvs_F_nontrue_CSIL_E20_20261019_r01/experiments/cvs_F_nontrue20/runner.py'
spec=importlib.util.spec_from_file_location('_F_nontrue_complete_analysis',runner);boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.bootstrap()
result=dict(summary=read(root/'analysis.json'),logs={},paired={},source_pairs={})
import torch
initials=[torch.load(root/(a+'-s2026092701')/'source/initial_smoke.pt',map_location='cpu',weights_only=False)['model'] for a in ('B0','C0','B1','M')]
result['initial_student_states_exact_equal']=all(set(q)==set(initials[0]) and all(torch.equal(v,q[k]) for k,v in initials[0].items()) for q in initials[1:])
if not result['initial_student_states_exact_equal']:raise ValueError('Same-seed scratch F initial states differ')
def records(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else p.open() as f:return [json.loads(s) for s in f if s.strip()]
def ppairs(y,b,m):return dict(corrected=int(((m==y)&(b!=y)).sum()),damaged=int(((m!=y)&(b==y)).sum()),both_correct=int(((m==y)&(b==y)).sum()),both_wrong=int(((m!=y)&(b!=y)).sum()),count=len(y),net_pp=float(100*((m==y).mean()-(b==y).mean())))
for arm in ('B0','C0','B1','M'):
 source=root/(arm+'-s2026092701')/'source';rr=records(source/'mechanism_steps.jsonl.gz');unit=records(source/'unit_steps.jsonl.gz');epochs=records(source/'metrics_epoch.jsonl');native=records(source/'step_metrics.jsonl.gz')
 if any(len(a)!=4440 for a in (rr,unit,native)) or len(epochs)!=20:raise ValueError('Missing complete training records')
 u=[v for v in native if v['step']>13*222]
 if len(u)!=7*222 or any('train/pseudo_truth_available' not in v['metrics'] or float(v['metrics']['train/pseudo_truth_available'] or 0)!=0. for v in u):raise ValueError('Complete hidden-U truth boundary differs')
 ema_exact=all(float(v['metrics'].get('train/a1_ema_successful_updates',-1))==v['step'] for v in native)
 if not ema_exact:raise ValueError('Full ownEMA successful update sequence differs')
 errors=[s for s in (root/'logs'/(arm+'-s2026092701.log')).read_text().splitlines() if re.search(r'Traceback|CUDA out of memory|FloatingPointError|^Killed$|RuntimeError|ModuleNotFoundError',s)]
 formula=[];curves=[]
 for r in rr:
  expected=r['ordinary_CE'] if arm=='B0' else .5*(r['ordinary_CE']+r['extra_CE'])+(.5*r['nontrue_KL'] if arm=='M' else 0)
  formula.append(abs(expected-r['total_L']))
  if r['L_forwards']!=(1 if arm=='B0' else 2) or r['teacher_packets']!=(0 if arm=='B0' else 128) or r['As_packets']!=(128 if arm in ('B1','M') else 0) or not math.isfinite(r['total_L']):raise ValueError('Actual execution/budget differs')
 for e in range(20):
  rs=rr[e*222:(e+1)*222];mean=lambda k:sum(r.get(k,0.) for r in rs)/222
  curves.append(dict(epoch=e+1,total_L=mean('total_L'),ordinary_CE=mean('ordinary_CE'),As_or_raw_CE=mean('extra_CE'),masked_nontrue=mean('nontrue_KL'),teacher_correct_fraction=sum(r.get('teacher_correct',0) for r in rs)/(128*222)))
 seqcounts={'L':0,'U':0};seqpackets={'L':0,'U':0}
 with gzip.open(source/'training_sequence.jsonl.gz','rt') as f:
  for line in f:
   r=json.loads(line);seqcounts[r['role']]+=1;seqpackets[r['role']]+=len(r['ids'])
 result['logs'][arm]=dict(complete_records=dict(mechanism=len(rr),native_steps=len(native),unit=len(unit),epochs=len(epochs)),hidden_U_steps=len(u),U_truth_available_max=0.,ownEMA_all_updates_exact=ema_exact,sequence_batches=seqcounts,sequence_packets=seqpackets,errors=errors,max_L_formula_error=max(formula),curves=curves,native_epochs=epochs,actual=read(source/'training_result.json'),source_completion=read(source/'completion.json'),prediction_completion=read(root/(arm+'-s2026092701')/'prediction/complete.json'),ordinary=read(source/'ordinary_augmentation_actual.json'))
paths=[root/(a+'-s2026092701')/'source/training_sequence.jsonl.gz' for a in ('B0','C0','B1','M')]
with gzip.open(paths[0],'rt') as a,gzip.open(paths[1],'rt') as b,gzip.open(paths[2],'rt') as cc,gzip.open(paths[3],'rt') as mm:
 same={'B0_C0':True,'C0_B1':True,'B1_M':True}
 for x in itertools.zip_longest(a,b,cc,mm):
  for key,i,j in [('B0_C0',0,1),('C0_B1',1,2),('B1_M',2,3)]:same[key]=same[key] and x[i]==x[j]
result['source_training_sequences_equal']=same
from experiments.cvs_early_geometry.evaluate import truth_arrays
with np.load(root/'M-s2026092701/prediction/predictions.npz',allow_pickle=False) as f:ids=f['ids'].copy();mp={v:f[v].copy() for v in VIEWS}
truth=read(Path(c['project'])/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json');y,rx,days=truth_arrays(truth,ids)
for a in ('B0','C0','B1'):
 with np.load(root/(a+'-s2026092701')/'prediction/predictions.npz',allow_pickle=False) as f:
  if not np.array_equal(ids,f['ids']):raise ValueError('Paired formal physical IDs differ')
  result['paired'][a]={v:ppairs(y,f[v],mp[v]) for v in VIEWS}
with np.load(root/'M-s2026092701/source_qualification/predictions.npz',allow_pickle=False) as f:sy=f['TX'].copy();sid=f['ids'].copy();sclean=f['clean'].argmax(1);ssat=f['satellite'].argmax(-1)
for a in ('B0','C0','B1'):
 with np.load(root/(a+'-s2026092701')/'source_qualification/predictions.npz',allow_pickle=False) as f:
  if not np.array_equal(sid,f['ids']):raise ValueError('Paired source IDs differ')
  result['source_pairs'][a]=dict(clean=ppairs(sy,f['clean'].argmax(1),sclean),urban={str(j):ppairs(np.tile(sy,2),f['satellite'][:,j].argmax(-1).reshape(-1),ssat[:,j].reshape(-1)) for j in (1,2)})
print(json.dumps(result))
'''
def main(runtime,output):
 rt=json.loads(Path(runtime).read_text(encoding='utf8'));out=Path(output);out.mkdir(parents=True,exist_ok=False)
 script=SCRIPT.replace('CONFIG',repr(dict(project=rt['project'],run_id=d.RUN))).replace('VIEWS',repr(d.VIEWS))
 x=remote(script,rt);(out/'complete_analysis.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8');render(x,out)
 print(json.dumps(dict(status=x['summary']['status'],differences=x['summary']['differences'],paired=x['paired'],source_sequences=x['source_training_sequences_equal'])))
def render(x,out):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors=dict(B0='#8a8a8a',C0='#c59342',B1='#315e8d',M='#148f7b');names=dict(B0='NoAs',C0='Raw CE control',B1='As CE control',M='As + nontrue');fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for a in colors:
  metrics=x['summary']['rows'][a]['overall'];cs=x['logs'][a]['curves'];e=[v['epoch'] for v in cs]
  axes[0,0].plot(range(7),[100*metrics[v]['accuracy'] for v in d.VIEWS],'-o',label=names[a],color=colors[a],ms=4)
  axes[0,1].plot(e,[v['ordinary_CE'] for v in cs],label=a,color=colors[a])
  if a!='B0':axes[1,0].plot(e,[v['As_or_raw_CE'] for v in cs],label=a,color=colors[a])
  if a!='B0':axes[1,1].plot(e,[v['masked_nontrue'] for v in cs],label=a+(': active (As)' if a=='M' else ': shadow ('+('raw' if a=='C0' else 'As')+')'),color=colors[a])
 axes[0,0].set(xticks=range(7),xticklabels=['clean','high','mid','low sub','high urb','mid urb','low urb'],ylabel='Accuracy (%)');axes[0,0].legend(frameon=False)
 axes[0,1].set(xlabel='Epoch',ylabel='Ordinary CE');axes[1,0].set(xlabel='Epoch',ylabel='Raw / As CE');axes[1,1].set(xlabel='Epoch',ylabel='Correct-masked nontrue auxiliary')
 axes[1,1].legend(frameon=False,fontsize=9)
 for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
 fig.savefig(out/'seven_views_and_full_training.png',dpi=180);fig.savefig(out/'seven_views_and_full_training.pdf');plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime');p.add_argument('--output',required=True);p.add_argument('--render-cache');a=p.parse_args()
 if a.render_cache:render(json.loads(Path(a.render_cache).read_text(encoding='utf8')),Path(a.output))
 else:main(a.runtime,a.output)
