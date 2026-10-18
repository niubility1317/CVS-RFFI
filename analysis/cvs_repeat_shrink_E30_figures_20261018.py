"""Export measured fixed-view accuracy and paired differences from public aggregates."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def run(data,out):
 views=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban');labels=('Clean','High\nsuburban','Mid\nsuburban','Low\nsuburban','High\nurban','Mid\nurban','Low\nurban');x=np.arange(7);fig,axes=plt.subplots(1,2,figsize=(13.6,4.5),gridspec_kw={'width_ratios':[1.25,1]});colors={'B00':'#65758B','B10':'#16877E'}
 for mode,shift in [('B00',-.18),('B10',.18)]:
  values=[100*data['summary'][mode][v]['accuracy'] for v in views];axes[0].bar(x+shift,values,width=.34,color=colors[mode],label=mode)
 axes[0].set_ylim(0,100);axes[0].set_ylabel('Default-head accuracy (%)');axes[0].set_title('Fixed E30 models: unchanged received IQ');axes[0].legend(frameon=False,ncol=2);axes[0].plot([5.6,6.4],[60,60],ls='--',c='#8C4560',lw=1);axes[0].text(6,62,'low urban >60%',ha='center',fontsize=8,c='#8C4560')
 delta=[data['paired']['B10__B00'][v]['delta_accuracy_pp'] for v in views];axes[1].bar(x,delta,color=['#16877E' if z>=0 else '#BA5554' for z in delta],width=.6);axes[1].axhline(0,c='#344258',lw=.9);axes[1].set_ylabel('B10 - B00 (percentage points)');axes[1].set_title('Same-budget independent IQ augmentation value')
 extent=max(.05,max(abs(z) for z in delta));axes[1].set_ylim(min(0,min(delta))-extent*.25,max(0,max(delta))+extent*.25)
 for i,z in enumerate(delta):axes[1].text(i,z+(extent*.035 if z>=0 else -extent*.035),f'{z:+.3f}',ha='center',va='bottom' if z>=0 else 'top',fontsize=8)
 for ax in axes:
  ax.set_xticks(x,labels);ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 fig.suptitle('B00/B10 only: no satellite training; one model seed; no Ag-As interaction',fontsize=11);fig.tight_layout(rect=[0,0,1,.96]);o=Path(out);o.mkdir(parents=True,exist_ok=True)
 for suffix in ('png','pdf'):fig.savefig(o/('repeat_shrink_E30_accuracy.'+suffix),dpi=180,bbox_inches='tight')
 plt.close(fig)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('input');a.add_argument('output');args=a.parse_args();run(json.loads(Path(args.input).read_text(encoding='utf8')),args.output);print(json.dumps(dict(status='FIGURES_SAVED',IQ=0,CP=0,NN=0,updates=0)))
