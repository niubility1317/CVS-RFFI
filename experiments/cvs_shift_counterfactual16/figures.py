"""Complete source default-head diagnostic figures, no training loss claim."""
from . import design as d
def main():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 a=d.read(d.ROOT/d.LOCAL/'final/analysis_summary.json');out=d.ROOT/d.LOCAL/'final';epochs=a['epochs'];x=[e['epoch'] for e in epochs]
 fig,axes=plt.subplots(3,1,figsize=(11,10),layout='constrained')
 axes[0].plot(x,[100*e['full_score']['accuracy'] for e in epochs],'-o',label='Full original ordinary');axes[0].plot(x,[100*e['bypass_score']['accuracy'] for e in epochs],'-s',label='Same draw: circular shift bypassed');axes[0].axhline(100*a['raw_score']['accuracy'],ls='--',color='#777777',label='Raw once');axes[0].set(ylabel='Fit-L default-head accuracy (%)',title='Actual native configure epochs 1..13; frozen model; zero updates');axes[0].legend(frameon=False)
 axes[1].bar(x,[e['paired']['nonzero_shift']['recovery_pp'] or 0 for e in epochs],color='#36789b');axes[1].axhline(2,ls='--',color='#777777',label='Preregistered pooled recovery criterion (2 pp)');axes[1].set(ylabel='Nonzero-shift recovery (pp)',title='All epochs retained; no epoch selection');axes[1].legend(frameon=False)
 axes[2].bar(x,[e['nonzero_shift'] for e in epochs],color='#b77c56',label='Actual nonzero shift');axes[2].plot(x,[e['changed'] for e in epochs],'k.-',label='Final IQ changed');axes[2].set(xlabel='Original schedule epoch',ylabel='Packet observations',xticks=x);axes[2].legend(frameon=False)
 fig.suptitle('F + cross-RX same-TX / own frozen B00 / fit L5040\nCircular-shift source counterfactual '+a['decision']+'; not Ag generalization efficacy');fig.savefig(out/'ordinary_shift_epoch_effects.png',dpi=170);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
 for ax,key in zip(axes,('by_RX','by_day','by_TX')):
  values=a['pooled'][key];ax.bar(list(values),[v['recovery_pp'] or 0 for v in values.values()],color='#36789b');ax.axhline(0,color='#555555',lw=.7);ax.set(title=key.replace('by_',''),ylabel='Nonzero-shift recovery (pp)')
  for i,v in enumerate(values.values()):ax.annotate('n='+str(v['count']),(i,v['recovery_pp'] or 0),xytext=(0,5 if (v['recovery_pp'] or 0)>=0 else -13),textcoords='offset points',ha='center',fontsize=8)
 z=a['pooled']['nonzero_shift'];fig.suptitle('Same-draw full vs shift-bypassed source identity / all13 schedules\nRepairs '+str(z['repairs'])+'; new breaks '+str(z['broken'])+'; repeated physical observations');fig.savefig(out/'ordinary_shift_strata.png',dpi=170);plt.close(fig);print('Saved two complete source diagnostic figures')
if __name__=='__main__':main()
