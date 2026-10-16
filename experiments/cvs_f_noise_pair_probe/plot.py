"""Standard scientific figures of the completed frozen derivative observer only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS=('#326b9c','#c36c35')
def draw(root,out):
 root=Path(root);out=Path(out);out.mkdir(exist_ok=False);r=json.loads((root/'analysis_summary.json').read_text());records=json.loads((root/'independent_gradient_records.json').read_text());strata=json.loads((root/'source_strata.json').read_text());seeds=list(r['rows']);groups=('default_head','SIG_TIME','Sinc_Hz','time_path','remaining')
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.bbox':'tight'})
 def save(fig,name):
  for extension in ('png','pdf'):fig.savefig(out/(name+'.'+extension))
  plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(13.5,4.5));ax=axes[0]
 for color,seed in zip(COLORS,seeds):
  rows=[v for v in records if str(v['model_seed'])==seed and v['group']=='whole'];ax.plot([v['batch']+1 for v in rows],[v['trace_ratio'] for v in rows],color=color,label=seed[-2:]+' (sum ratio %.3f)'%r['rows'][seed]['diagnostic']['trace_ratio'])
 ax.axhline(.8,color='black',linestyle='--',linewidth=1,label='Fixed rationale threshold');ax.set(xlabel='Source batch (last batch n=48)',ylabel='Antithetic / independent-pair variance',title='Whole default-head CE gradient');ax.legend(fontsize=9)
 x=np.arange(len(groups))
 for offset,color,seed in zip((-.18,.18),COLORS,seeds):
  vals=[]
  for g in groups:
   rows=[v for v in records if str(v['model_seed'])==seed and v['group']==g];vals.append(sum(v['antithetic_pair_trace'] for v in rows)/sum(v['independent_pair_trace'] for v in rows))
  bars=axes[1].bar(x+offset,vals,.35,color=color,label=seed[-2:]);axes[1].bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
 axes[1].set(xticks=x,xticklabels=groups,ylabel='Summed variance ratio',title='All five disjoint parameter groups');axes[1].tick_params(axis='x',rotation=25);axes[1].axhline(1,color='gray',linewidth=1);axes[1].legend(title='Model seed')
 fig.suptitle('F + cross-RX same-TX | frozen E20 default heads | fit L5040 | zero updates');fig.tight_layout();save(fig,'noise_gradient_variance')
 fig,axes=plt.subplots(1,2,figsize=(12.5,4.4))
 for color,seed in zip(COLORS,seeds):
  for sign,style in ((1,'-'),(-1,'--')):
   vals=[100*r['rows'][seed]['scores'][f'draw{k}_sign{sign}']['accuracy'] for k in range(8)];axes[0].plot(range(1,9),vals,style,marker='o',color=color,label=f'{seed[-2:]} sign {sign:+d}')
 axes[0].set(xlabel='Independent primitive noise draw',ylabel='Fit-L accuracy (%)',title='Descriptive mixture of six practical scenes');axes[0].legend(fontsize=9)
 x=np.arange(len(seeds));ratios=[r['rows'][s]['diagnostic']['trace_ratio'] for s in seeds];neg=[r['rows'][s]['diagnostic']['negative_covariance_fraction'] for s in seeds]
 axes[1].bar(x-.18,100*np.array(ratios),.35,label='Variance ratio (%)',color='#467e9e');axes[1].bar(x+.18,100*np.array(neg),.35,label='Negative-covariance batches (%)',color='#8a9d55');axes[1].set(xticks=x,xticklabels=[s[-2:] for s in seeds],xlabel='Model/global ordinary seed',ylabel='Percent',ylim=(0,110),title='Both-seed fixed rationale criteria');axes[1].legend(fontsize=9)
 fig.suptitle('No new model training, held-RX or target result; repeated noise views are not new physical samples');fig.tight_layout();save(fig,'source_noise_readout')
 fig,axes=plt.subplots(1,2,figsize=(12,4.3))
 for ax,dim,values in zip(axes,('state','lock'),(('LOS','PARTIAL','BLOCKED'),('locked','degraded','unlocked'))):
  x=np.arange(len(values))
  for offset,color,seed in zip((-.18,.18),COLORS,seeds):
   rows=[next(v for v in strata if str(v['model_seed'])==seed and v['dimension']==dim and v['value']==value) for value in values];bars=ax.bar(x+offset,[100*v['accuracy'] for v in rows],.35,label=seed[-2:],color=color);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
  ax.set(xticks=x,xticklabels=values,ylabel='Repeated fit-L decision accuracy (%)',title=dim,ylim=(0,100));ax.legend(title='Model seed')
 fig.suptitle('Frozen default heads | all eight positive and negative noise views | state association only');fig.tight_layout();save(fig,'source_noise_strata')
 return dict(status='VERIFIED',new_NN=0,updates=0,source_only=True,figures=[p.name for p in out.iterdir()])
