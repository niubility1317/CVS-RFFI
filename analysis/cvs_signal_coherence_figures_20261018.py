"""Plot saved source aggregates only; no IQ, checkpoint, prediction or training."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def run(source,output):
 r=json.loads(Path(source).read_text(encoding='utf8'));out=Path(output);out.mkdir(parents=True,exist_ok=False)
 if r['status']!='SOURCE_PERIODIC_SHAPE_READOUT_COMPLETE' or len(r['folds'])!=18 or r['source_L_unique']!=6300 or r['source_V_unique']!=27000:raise ValueError('Complete source-only aggregate required')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
 fig,axes=plt.subplots(1,3,figsize=(14,4.4),constrained_layout=True);rx=list(r['per_RX']);values=[100*r['per_RX'][x]['risk_gain'] for x in rx];axes[0].bar(range(5),values,color='#2166ac');axes[0].set_xticks(range(5),rx);axes[0].set_xlabel('Source RX');axes[0].set_ylabel('V shape SSE reduction (%)');axes[0].set_ylim(0,100);axes[0].set_title('Other TX/day L -> V; all five RX');axes[0].grid(axis='y',alpha=.2)
 risk=np.array([100*x['risk_gain'] for x in r['folds']]).reshape(3,6);im=axes[1].imshow(risk,vmin=0,vmax=100,cmap='Blues',aspect='auto');axes[1].set_xticks(range(6),range(6));axes[1].set_yticks(range(3),(1,2,3));axes[1].set_xlabel('Held TX');axes[1].set_ylabel('Held day');axes[1].set_title('Statistical folds; no training RX held out')
 for i in range(3):
  for j in range(6):axes[1].text(j,i,f'{risk[i,j]:.1f}',ha='center',va='center',color='white' if risk[i,j]>60 else 'black',fontsize=9)
 fig.colorbar(im,ax=axes[1],shrink=.8,label='V SSE reduction (%)');ratio=r['shared_RX_SSE']/r['TXconditional_SSE'];axes[2].bar((0,1),(1,ratio),color=('#555555','#d6604d'));axes[2].axhline(1.05,color='black',ls='--',label='Fixed 1.05 limit');axes[2].set_xticks((0,1),('Same-TX L','Other-TX L'),rotation=15);axes[2].set_ylabel('Residual SSE / same-TX control');axes[2].set_title('Shared additive contrast qualification failed');axes[2].legend(fontsize=9);axes[2].text(1,ratio,f'{ratio:.3f}',ha='center',va='bottom');axes[2].set_ylim(0,ratio*1.2);axes[2].grid(axis='y',alpha=.2)
 fig.suptitle('Source association only; no IQ augmentation or identity-recognition result',fontsize=11)
 for ext in ('png','pdf'):fig.savefig(out/('source_shape_transfer.'+ext))
 plt.close(fig);return dict(status='FIGURES_WRITTEN',files=2,IQ=0,CP=0,NN=0,updates=0,target=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.source,a.output)))
