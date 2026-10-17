"""Source-only static summaries;no model,IQ,checkpoint or target reads."""
from pathlib import Path
import numpy as np

def make(out,rows,metrics):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=Path(out);figures=[]
 def save(fig,name):
  fig.tight_layout()
  for suffix in ('png','pdf'):fig.savefig(out/(name+'.'+suffix),dpi=180,bbox_inches='tight')
  plt.close(fig);figures.append(dict(name=name,png=str(out/(name+'.png')),pdf=str(out/(name+'.pdf'))))
 def grad(obj,draw,scene,cond,name='head_W',kind='parameter'):
  r=[v for v in rows if v['objective']==obj and v['draw']==draw and v['scene']==scene and v['condition']==cond and v['name']==name and v['kind']==kind and v['dimension']=='overall']
  if len(r)!=1:raise ValueError('One complete source gradient cell required')
  return r[0]
 scenes=('practical_mid_urban','practical_low_urban');conditions=('full','no_noise/fixed','no_noise/total');colors=('#425e9a','#dd8e2c','#35956c');labels=('full','noise removed / fixed RX','noise removed / RX recomputed')
 fig,axes=plt.subplots(1,2,figsize=(11,4.6));x=np.arange(4);ticks=['mid d0','mid d1','low d0','low d1']
 for ci,cond in enumerate(conditions):
  values=[100*grad('margin',draw,scene,cond)['negative_fraction_all'] for scene in scenes for draw in (0,1)]
  axes[0].bar(x+(ci-1)*.25,values,width=.24,color=colors[ci],label=labels[ci])
  a=[100*next(v for v in metrics if v['scene']==scene and v['draw']==draw and v['condition']==cond and v['dimension']=='overall')['accuracy'] for scene in scenes for draw in (0,1)]
  axes[1].bar(x+(ci-1)*.25,a,width=.24,color=colors[ci],label=labels[ci])
 for ax in axes:ax.set_xticks(x,ticks);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axes[0].set(ylabel='Negative clean/view W-gradient pairs (% of all 900)',title='Default-head margin-gradient direction',ylim=(0,100));axes[1].set(ylabel='Source L accuracy (%)',title='Frozen default head; no updates',ylim=(0,100));axes[1].legend(fontsize=8,loc='lower left');fig.suptitle('L-only paired noise counterfactual; one model seed, two channel draws')
 save(fig,'paired_noise_gradient_and_accuracy')
 fig,axes=plt.subplots(1,3,figsize=(13,4.3),sharey=True)
 for ax,obj in zip(axes,('margin','CE0','CE01')):
  for ci,cond in enumerate(conditions):ax.bar(x+(ci-1)*.25,[100*grad(obj,draw,scene,cond)['negative_fraction_all'] for scene in scenes for draw in (0,1)],width=.24,color=colors[ci],label=labels[ci])
  ax.set_xticks(x,ticks);ax.set_title(obj);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_ylim(0,100)
 axes[0].set_ylabel('Negative pairs (% of all 900)');axes[-1].legend(fontsize=8,loc='upper left');fig.suptitle('Calibration control: margin is primary; CE0 and LS=.01 are descriptive')
 save(fig,'objective_calibration_control')
 return figures
