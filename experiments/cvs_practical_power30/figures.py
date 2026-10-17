"""Standalone source curves and matched frozen seven-view scores;no inference."""
from pathlib import Path
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
LABELS=('Clean','Sub high','Sub mid','Sub low','Urb high','Urb mid','Urb low')

def render(data,parent,out):
 import matplotlib
 matplotlib.use('Agg')
 from matplotlib import pyplot as plt
 import numpy as np
 if len(parent['epoch_metrics'])!=20:raise ValueError('Complete source parent20 required')
 for row in data['rows'].values():
  if len(row['epoch_metrics'])!=10 or row['step_metrics']['records']!=2220 or row['prediction']['views']!=list(VIEWS):raise ValueError('Complete fixed E20+10 rows required')
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axes=plt.subplots(2,2,figsize=(14,8.6),layout='constrained');colors={'QUALITY_POWER':'#2563a6','NOMINAL_AS':'#b7524c'};summary=data['result']['summary']
 ax=axes[0,0]
 for name,row in data['rows'].items():
  y=[r['step_metrics']['epochs'][str(e)]['train/loss']['mean'] for r,n in ((parent,20),(row,10)) for e in range(1,n+1)]
  ax.plot(range(1,31),y,color=colors[name],lw=1.8,label=name)
 ax.axvline(20.5,color='0.4',ls='--',lw=1);ax.text(20.7,.06,'Optimizer reset',transform=ax.get_xaxis_transform(),fontsize=8)
 ax.set(title='A  All source training steps: fixed E20 + 10',xlabel='Cumulative epoch',ylabel='Mean total training objective',xlim=(1,30));ax.legend(frameon=False,fontsize=9)
 ax=axes[0,1];xx=np.arange(7)
 for name in colors:ax.plot(xx,[100*summary[name][v]['accuracy'] for v in VIEWS],color=colors[name],marker='o',lw=1.8,label=name)
 ax.set_xticks(xx,LABELS,rotation=25,ha='right');ax.set(title='B  Frozen default head on unchanged received IQ',ylabel='Accuracy (%)',ylim=(0,100));ax.legend(frameon=False,fontsize=9)
 ax=axes[1,0];delta=[data['result']['paired_differences'][v]['delta_accuracy_pp'] for v in VIEWS];f1=[100*(summary['QUALITY_POWER'][v]['macro_F1']-summary['NOMINAL_AS'][v]['macro_F1']) for v in VIEWS]
 ax.bar(xx,delta,color='#2563a6',width=.6,label='Accuracy difference');ax.plot(xx,f1,color='#db8b25',marker='s',lw=1.5,label='Macro F1 difference');ax.axhline(0,color='0.25',lw=.8)
 for i,x in enumerate(delta):ax.annotate(f'{x:+.3f}',(i,x),xytext=(0,5 if x>=0 else -12),textcoords='offset points',ha='center',fontsize=8)
 ax.set_xticks(xx,LABELS,rotation=25,ha='right');ax.set(title='C  Quality-power minus exact nominal As control',ylabel='Difference (pp)');ax.margins(y=.25);ax.legend(frameon=False,fontsize=9)
 ax=axes[1,1]
 for j,name in enumerate(colors):
  gap=[100*(summary[name]['clean']['accuracy']-summary[name][v]['accuracy']) for v in VIEWS[1:4]];pos=np.arange(3)+(j-.5)*.32
  ax.bar(pos,gap,color=colors[name],width=.32,label=name)
  for x,y in zip(pos,gap):ax.text(x,y+.15,f'{y:.2f}',ha='center',fontsize=8)
 ax.axhline(2,color='0.3',ls='--',lw=1,label='Exploratory gap 2 pp');ax.set_xticks(range(3),LABELS[1:4]);ax.set(title='D  Each suburban gap to the same model clean',ylabel='Clean minus suburban (pp)');ax.margins(y=.22);ax.legend(frameon=False,fontsize=8)
 for ax in axes.flat:ax.grid(axis='y',color='0.9',lw=.6);ax.set_axisbelow(True)
 fig.suptitle('Finite-window receiver quality | full source | seed 2026092701 | matched E20 + 10',fontsize=14)
 stem=out/'quality_power_matched_E30';fig.savefig(stem.with_suffix('.png'),dpi=180);fig.savefig(stem.with_suffix('.pdf'));plt.close(fig)
 return dict(status='PLOTS_WRITTEN',PNG=1,PDF=1,IQ=0,CP=0,NN=0,new_updates=0)
