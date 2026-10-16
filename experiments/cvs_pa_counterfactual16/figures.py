"""Scientific source figures from independently recounted fixed matrix."""
from . import design as d
def main():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 a=d.read(d.ROOT/d.LOCAL/'final/analysis_summary.json');out=d.ROOT/d.LOCAL/'final';epochs=a['epochs'];x=[e['epoch'] for e in epochs]
 fig,axes=plt.subplots(3,1,figsize=(11,10),layout='constrained')
 axes[0].plot(x,[100*e['full_score']['accuracy'] for e in epochs],'-o',label='Full original ordinary');axes[0].plot(x,[100*e['bypass_score']['accuracy'] for e in epochs],'-s',label='Same draw: PA only bypassed');axes[0].axhline(100*a['raw_score']['accuracy'],ls='--',color='#777777',label='Raw once');axes[0].set(ylabel='Fit-L default-head accuracy (%)',title='All20 original augmentation schedules; zero training');axes[0].legend(frameon=False)
 axes[1].bar(x,[e['paired']['changed']['recovery_pp'] or 0 for e in epochs],color='#36789b');axes[1].axhline(2,ls='--',color='#777777',label='Preregistered pooled recovery criterion (2 pp)');axes[1].set(ylabel='PA-changed packet recovery (pp)',title='Epoch effects reported separately; no epoch selection');axes[1].legend(frameon=False)
 axes[2].bar(x,[e['triggered'] for e in epochs],color='#b77c56',label='Actual PA triggered');axes[2].plot(x,[e['changed'] for e in epochs],'k.-',label='Actual IQ changed');axes[2].set(xlabel='Original schedule epoch',ylabel='Packet observations',xticks=x);axes[2].legend(frameon=False)
 fig.suptitle('F+CROSS_RX / frozen own B00 / source fit L5040\nPA counterfactual '+a['decision']+'; not Ag generalization efficacy');fig.savefig(out/'ordinary_PA_epoch_effects.png',dpi=170);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
 for ax,key in zip(axes,('by_RX','by_day','by_TX')):
  v=a['pooled'][key];labels=list(v);ax.bar(labels,[s['recovery_pp'] or 0 for s in v.values()],color='#36789b');ax.axhline(0,color='#555555',lw=.7);ax.set(title=key.replace('by_',''),ylabel='PA-changed recovery (pp)');
  for i,s in enumerate(v.values()):ax.annotate('n='+str(s['count']),(i,s['recovery_pp'] or 0),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
 fig.suptitle('Source same-draw PA-changed identity recovery / all20 schedules\nRepairs '+str(a['pooled']['changed']['repairs'])+'; new breaks '+str(a['pooled']['changed']['broken'])+'; repeated physical packets, no independent-packet significance claim');fig.savefig(out/'ordinary_PA_strata.png',dpi=170);plt.close(fig);print('Saved two complete source figures')
if __name__=='__main__':main()
