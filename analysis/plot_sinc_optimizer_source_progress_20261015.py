"""Plot all saved complete source epochs; no IQ/checkpoints/query or target scores."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main(src,out):
 v=json.loads(Path(src).read_text(encoding='utf8'));assert v['source_only'] and not v['query_read'] and not v['U_truth_read']
 assert len(v['rows'])==2;out=Path(out)
 if out.exists():raise FileExistsError('Preserve prior snapshot')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(11,4.2),dpi=180)
 for row,color in zip(v['rows'],['#287da1','#279578']):
  records=row['source_curves'];assert [r['epoch'] for r in records]==list(range(1,row['completed_epochs']+1));x=[r['epoch'] for r in records]
  label=row['row_id'].split('-s')[0]
  axes[0].plot(x,[r['train_loss_tx_labeled'] for r in records],lw=1.8,color=color,label=label)
  axes[1].plot(x,[r['val_tx_acc'] for r in records],lw=1.8,color=color)
 axes[0].set(title='Native L identity cross-entropy',ylabel='Cross-entropy',ylim=(0,None))
 axes[1].set(title='Source V default-head accuracy',ylabel='Accuracy (%) - native percent unit',ylim=(85,100))
 for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.16);ax.set_xlim(1,max(r['completed_epochs'] for r in v['rows'])+.3)
 fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.055))
 fig.suptitle('Fixed SIG_TIME | optimizer-only Sinc units | source snapshot',x=.07,y=.98,ha='left',fontsize=14,fontweight='bold')
 fig.text(.07,.014,'Seed 2026092701; all complete source epochs shown. Fixed 50 epochs; original SSL starts at E34. Source health is not target gain.',fontsize=8,color='#555555')
 fig.tight_layout(rect=(0,.13,1,.91));fig.savefig(out);plt.close(fig)
 print(json.dumps(dict(status='VERIFIED',complete_epochs=[r['completed_epochs'] for r in v['rows']],source_only=True,query_read=False)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();main(a.source,a.output)
