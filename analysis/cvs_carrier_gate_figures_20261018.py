"""Export full matrix and complete training curves from public recounted metadata."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MODES=('NOAS_BATCH','NOAS_PACKET','AS_BATCH','AS_PACKET')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
LABELS=('Clean','High sub','Mid sub','Low sub','High urban','Mid urban','Low urban')
COLORS=('#73808b','#4a91bf','#d49952','#ad493d')
def run(folder):
 p=Path(folder);v=json.loads((p/'result.json').read_text(encoding='utf8'));s=json.loads((p/'source_analysis.json').read_text(encoding='utf8'));plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained');x=list(range(7))
 for i,m in enumerate(MODES):axes[0].plot(x,[100*v['summary'][m][z]['accuracy'] for z in VIEWS],marker='o',label=m,color=COLORS[i])
 axes[0].set(xticks=x,xticklabels=LABELS,ylabel='Accuracy (%)',title='Fixed E30 / common NoAs20 parent / default head / one model seed');axes[0].legend(ncol=2);axes[0].grid(axis='y',alpha=.2)
 a=v['paired']['AS_PACKET__AS_BATCH'];b=v['paired']['NOAS_PACKET__NOAS_BATCH'];axes[1].bar([z-.18 for z in x],[a[z]['delta_accuracy_pp'] for z in VIEWS],width=.36,label='Packet - batch, As',color=COLORS[3]);axes[1].bar([z+.18 for z in x],[b[z]['delta_accuracy_pp'] for z in VIEWS],width=.36,label='Packet - batch, no As',color=COLORS[1]);axes[1].axhline(0,color='black',lw=.8);axes[1].set(xticks=x,xticklabels=LABELS,ylabel='Paired accuracy difference (pp)',title='Training organization only; no new IQ Ag');axes[1].legend();axes[1].grid(axis='y',alpha=.2)
 for ext in ('png','pdf'):fig.savefig(p/('accuracy_matrix.'+ext),dpi=180)
 plt.close(fig);fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for i,m in enumerate(MODES):
  r=s['rows'][m];ep=r['epoch_curves'];ref=r['reference_epoch_curves'];x=[20+z['epoch'] for z in ref]
  axes[0,0].plot(x,[z['total_L']['mean'] for z in ref],label=m,color=COLORS[i]);axes[0,1].plot(x,[z['common_CE']['mean'] for z in ref],color=COLORS[i]);axes[1,0].plot([20+z.get('epoch',j+1) for j,z in enumerate(ep)],[z['val_tx_acc'] for z in ep],color=COLORS[i]);axes[1,1].plot(x,[z['extra_CE']['mean'] for z in ref],color=COLORS[i])
 for ax,title in zip(axes.flat,['Full original L objective','Ordinary CE (unweighted)','Full source V27000 clean','Secondary CE (unweighted)']):ax.set(xlabel='Cumulative epoch',title=title);ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);axes[1,0].set_ylabel('Accuracy (%)')
 for ext in ('png','pdf'):fig.savefig(p/('complete_training_curves.'+ext),dpi=180)
 plt.close(fig);return dict(status='FIGURES_WRITTEN',figures=2,files=4,IQ=0,CP=0,NN=0,updates=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();print(json.dumps(run(a.folder)))
