"""Offline aggregate figures;zero IQ/checkpoint/network access."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(root):
 root=Path(root);a=json.loads((root/'independent_analysis.json').read_text());saved=np.load(root/'nonDC-line-fitRX3_4_6_8/source_line_statistics.npz',allow_pickle=False);prom=saved['cell_prominence_db'];bins=saved['shifted_bins'];out=root/'figures';out.mkdir(exist_ok=False)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
 colors=['#2878b5','#d94801','#238b45','#7b3294'];freq=(np.arange(256)-128)*25/256
 fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained',sharex=True)
 for w,ax in enumerate(axes):
  for r,rx in enumerate((3,4,6,8)):
   mean=np.full(256,np.nan);std=np.full(256,np.nan);mean[bins]=prom[w,:,:,r].mean((0,1));std[bins]=prom[w,:,:,r].std((0,1));ax.plot(freq,mean,color=colors[r],label='RX'+str(rx),lw=1.3);ax.fill_between(freq,mean-std,mean+std,color=colors[r],alpha=.07)
  ax.axvspan(-8*25/256,8*25/256,color='gray',alpha=.12,label='Excluded DC region');ax.axvline((a['modal_shifted_bin']-128)*25/256,color='#333333',ls=':',lw=1);ax.set(ylabel='Local excess (dB)',title=('Rectangular','Hann')[w]+' | mean +/- SD across18 TX/day cells');ax.grid(alpha=.17);ax.set_xlim(-12.5,12.5)
 axes[0].legend(ncol=5,loc='upper right',fontsize=9);axes[-1].set_xlabel('Frequency relative to baseband center (MHz)');fig.suptitle('Source L5040: non-DC local spectral excess | no identity model',fontsize=13)
 for ext in ('png','pdf'):fig.savefig(out/('source_local_spectral_profiles.'+ext))
 plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');folds=a['folds'];labels=[str(f['TX_index'])+'/'+str(f['day']) for f in folds];xx=np.arange(18)
 choices=np.asarray([f['selected_frequency_hz']/1e6 for f in folds]).reshape(6,3);im=axes[0,0].imshow(choices,aspect='auto',cmap='viridis');fig.colorbar(im,ax=axes[0,0],label='Selected MHz');axes[0,0].set(xticks=range(3),xticklabels=(1,2,3),yticks=range(6),xlabel='Held day',ylabel='Held TX index',title='Rectangular fit-only choice (allRX4)')
 for t in range(6):
  for dd in range(3):axes[0,0].text(dd,t,f'{choices[t,dd]:.2f}',ha='center',va='center',color='white' if choices[t,dd]<-5.5 else 'black',fontsize=9)
 for w,name in enumerate(('Rectangular','Hann')):
  axes[0,1].plot(xx,[f['held_RX_contrast_db'][w] for f in folds],marker='o',ms=3,label=name)
  axes[1,0].plot(xx,[100*(1-f['SSE'][w]/f['zero_SSE'][w]) for f in folds],marker='o',ms=3,label=name)
 axes[0,1].axhline(3,color='black',ls='--',lw=1,label='Fixed3dB');axes[0,1].set(ylabel='Held selected-RX contrast (dB)',title='Same fit-selected frequency, two windows');axes[0,1].legend(fontsize=9)
 axes[1,0].axhline(0,color='black',lw=1);axes[1,0].set(ylabel='Risk reduction vs zero RX effect (%)',title='All18 held TX/day folds');axes[1,0].legend(fontsize=9)
 for ax in (axes[0,1],axes[1,0]):ax.set_xticks(xx,labels,rotation=60);ax.set_xlabel('Held TX index/day');ax.grid(alpha=.15)
 axes[1,1].barh([0,1],[a['successful_folds'],a['modal_count']],color=['#238b45','#c0392b']);axes[1,1].axvline(15,color='black',ls='--',label='Fixed15/18');axes[1,1].set(yticks=[0,1],yticklabels=['Two-window line success','Exact modalRX/bin'],xlim=(0,18.6),xlabel='Number of folds',title='Predeclared decision: FAIL')
 for i,n in enumerate((a['successful_folds'],a['modal_count'])):axes[1,1].text(n+.1,i,str(n)+'/18',va='center')
 axes[1,1].legend(loc='lower right',fontsize=9);fig.suptitle('Source statistic only: reproducible RX contrast, failed fixed-frequency criterion',fontsize=12)
 for ext in ('png','pdf'):fig.savefig(out/('all_crossfit_folds_and_failure.'+ext))
 plt.close(fig);result=dict(status='PLOTS_WRITTEN',model_forwards=0,IQ_reads=0,checkpoint_reads=0,files=[dict(name=p.name,bytes=p.stat().st_size) for p in out.iterdir()]);(root/'plot_execution.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);arg=parser.parse_args();print(json.dumps(plot(arg.input)))
