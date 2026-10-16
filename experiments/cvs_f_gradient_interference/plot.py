"""Plot independently recomputed source scalars; never imports a model or IQ reader."""
import argparse,csv,json
from pathlib import Path

def draw(analysis,output):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 from matplotlib.colors import LogNorm
 analysis=Path(analysis);output=Path(output);output.mkdir(parents=True,exist_ok=False)
 with (analysis/'gradient_pairs.csv').open(encoding='utf-8',newline='') as f:pairs=list(csv.DictReader(f))
 report=json.loads((analysis/'analysis_summary.json').read_text(encoding='utf-8'));order=[(m,r) for m in ('C0','C1') for r in (0,1)];colors={'same_ordinary':'#377eb8','As':'#e07a29'}
 fig,axes=plt.subplots(1,3,figsize=(14,4.6),constrained_layout=True)
 for ax,key in zip(axes[:2],('norm_ratio','cosine')):
  for j,kind in enumerate(('same_ordinary','As')):
   data=[[float(v[key]) for v in pairs if v['group']=='whole' and v['checkpoint']==m and int(v['rep'])==r and v['secondary']==kind] for m,r in order]
   bp=ax.boxplot(data,positions=np.arange(4)+(j-.5)*.27,widths=.23,patch_artist=True,showfliers=True,medianprops={'color':'black','linewidth':1.3})
   for patch in bp['boxes']:patch.set_facecolor(colors[kind]);patch.set_alpha(.75)
   ax.plot([],[],color=colors[kind],lw=6,label='Repeated ordinary' if kind=='same_ordinary' else 'Full As')
  ax.set_xticks(range(4),[f'{m} / stream {r}' for m,r in order],rotation=18);ax.legend(fontsize=8);ax.set_xlabel('Frozen checkpoint / observation stream')
 axes[0].set_yscale('log');axes[0].axhline(1,color='grey',ls='--',lw=.9);axes[0].set_ylabel('Secondary / primary gradient norm')
 axes[1].axhline(0,color='grey',ls='--',lw=.9);axes[1].set_ylim(-1.02,1.02);axes[1].set_ylabel('Gradient cosine (physical Sinc coordinates)')
 for j,kind in enumerate(('same_ordinary','As')):
  values=[100*next(v for v in report['gradient_summary'] if v['checkpoint']==m and v['rep']==r and v['secondary']==kind and v['group']=='whole')['opposed_fraction'] for m,r in order]
  bars=axes[2].bar(np.arange(4)+(j-.5)*.3,values,width=.28,color=colors[kind],label='Repeated ordinary' if kind=='same_ordinary' else 'Full As')
  axes[2].bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
 axes[2].set_xticks(range(4),[f'{m} / stream {r}' for m,r in order],rotation=18);axes[2].set_ylim(0,55);axes[2].set_ylabel('Negative dot-product batches (%)');axes[2].legend(fontsize=8)
 fig.suptitle('Frozen source fit-L: gradient relationship, not an accuracy or AdamW influence estimate',fontsize=12)
 for ext in ('png','pdf'):fig.savefig(output/('gradient_interference.'+ext),dpi=170)
 plt.close(fig)
 groups=['default_head','SIG_TIME','Sinc_Hz','time_path','remaining'];values=np.array([[next(v for v in report['gradient_summary'] if v['checkpoint']==m and v['rep']==r and v['secondary']=='As' and v['group']==g)['median_norm_ratio'] for g in groups] for m,r in order],dtype=float)
 fig,ax=plt.subplots(figsize=(9,4.4),constrained_layout=True);im=ax.imshow(values,norm=LogNorm(vmin=max(.01,float(np.nanmin(values))),vmax=float(np.nanmax(values))),cmap='YlOrRd',aspect='auto')
 ax.set_xticks(range(5),['Default head','SIG_TIME','Sinc (Hz)','Time path','Remaining']);ax.set_yticks(range(4),[f'{m} / stream {r}' for m,r in order]);ax.set_title('All fixed parameter groups: median As / primary gradient norm ratio')
 for i in range(4):
  for j in range(5):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='black',fontsize=11)
 fig.colorbar(im,ax=ax,label='Norm ratio (log color scale)')
 for ext in ('png','pdf'):fig.savefig(output/('parameter_groups.'+ext),dpi=170)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4.7),constrained_layout=True);views=['primary_ordinary','same_ordinary','As']
 for m,r in order:
  vals=[100*next(v for v in report['source_decisions'] if v['checkpoint']==m and v['rep']==r and v['view']==view)['accuracy'] for view in views]
  ax.plot(range(3),vals,marker='o',ls='-' if r==0 else '--',color='#377eb8' if m=='C0' else '#e07a29',label=f'{m} / stream {r}')
 ax.set_xticks(range(3),['Primary ordinary','Repeated ordinary','Full As']);ax.set_ylim(0,100);ax.set_ylabel('Fit-L default-head accuracy (%)');ax.legend();ax.set_title('Descriptive stochastic train-mode readout of already-trained samples')
 fig.suptitle('One model seed; 0 new training; two observation streams are not independent model seeds',fontsize=10)
 for ext in ('png','pdf'):fig.savefig(output/('source_fit_readout.'+ext),dpi=170)
 plt.close(fig)
 return ['gradient_interference.png','parameter_groups.png','source_fit_readout.png']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(draw(a.analysis,a.output)))
