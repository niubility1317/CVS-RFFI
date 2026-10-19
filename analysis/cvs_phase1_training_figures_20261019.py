"""Export fixed-run training curves and seven-view metrics from aggregate JSON only."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def render(logs,metrics,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':180});colors={'T1':'#295A83','T6':'#C06C37','KD':'#6B4E9B'}
 fig,axs=plt.subplots(1,3,figsize=(12,3.8))
 for arm,v in logs['rows'].items():
  ep=v['epochs'];x=[r['epoch'] for r in ep]
  for ax,key in zip(axs,('native_loss','As_native_NLL','As_per_example_cosine_gradient_proxy')):
   ax.plot(x,[r.get(key,np.nan) for r in ep],label=arm,color=colors.get(arm));ax.set_xlabel('Cumulative epoch')
 for ax,title in zip(axs,('Full training objective (different across arms)','As native-scale smoothed CE (0.01)','As per-example cosine gradient proxy')):
  ax.set_title(title,loc='left',fontsize=9);ax.axvline(13.5,color='#aaa',lw=.7,ls=':');ax.axvline(20.5,color='#aaa',lw=.7,ls='--');ax.legend(frameon=False)
 fig.text(.07,.01,'All completed records parsed. Dotted: U starts; dashed: original optimizer/cosine reset. Gradient proxy excludes outer weights.',fontsize=8,color='#555');fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'training_curves.png',bbox_inches='tight');fig.savefig(out/'training_curves.pdf',bbox_inches='tight');plt.close(fig)
 if metrics is not None:
  views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'];labels=['Clean','High','Mid','Low suburban','High urban','Mid urban','Low urban'];fig,ax=plt.subplots(figsize=(10,4))
  for arm,byview in metrics.items():
   ax.plot(range(7),[100*byview[v]['accuracy'] for v in views],marker='o',label=arm,color=colors.get(arm))
  ax.set_xticks(range(7),labels,rotation=12);ax.set_ylabel('Frozen default-head accuracy (%)');ax.set_ylim(0,100);ax.legend(frameon=False);ax.set_title('Fixed clean and six practical observations',loc='left');fig.tight_layout();fig.savefig(out/'formal_seven_views.png',bbox_inches='tight');fig.savefig(out/'formal_seven_views.pdf',bbox_inches='tight');plt.close(fig)
 return {'status':'FIGURES_EXPORTED','files':[p.name for p in out.iterdir()]}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--logs',required=True);p.add_argument('--metrics');p.add_argument('--output',required=True);a=p.parse_args();x=render(json.loads(Path(a.logs).read_text()),json.loads(Path(a.metrics).read_text()) if a.metrics else None,a.output);print(json.dumps(x))
