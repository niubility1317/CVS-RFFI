"""Plot full-log C2/S4 mechanism evidence and fixed target score differences."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 summary=json.loads((root/'analysis/full_analysis.json').read_text(encoding='utf-8'))
 colors={'C2':'#b55129','S4':'#176d9a'}
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.grid':True,'grid.alpha':.16})
 fig,ax=plt.subplots(2,3,figsize=(15,8.3),layout='constrained')
 for row in ['C2','S4']:
  epochs=[json.loads(l) for l in (root/row/'logs.jsonl').read_text().splitlines()]
  actions=[json.loads(l) for l in (root/row/'game_actions.jsonl').read_text().splitlines()]
  audits=[json.loads(l) for l in (root/row/'game_audit.jsonl').read_text().splitlines()]
  grouped=defaultdict(list)
  for x in actions:grouped[x['epoch']].append(x)
  xs=np.arange(1,201);color=colors[row]
  ax[0,0].plot(xs,[sum(x['satellite_count'] for x in grouped[e])/sum(x['sample_count'] for x in grouped[e]) for e in xs],label=row,color=color)
  level=0.;events={x['step']:x['level'] for x in summary[row]['curriculum_events'] if x['changed']}
  if row=='C2':
   levels=[]
   for x in actions:
    levels.append(level);level=events.get(x['step'],level)
   ax[0,1].step(np.arange(9800)/49+1,levels,where='post',color=color,label='C2 internal level')
  ax[0,2].plot([x['epoch'] for x in audits],[x['recovered']['ce'] for x in audits],'-o',markersize=3,color=color,label=row+' recovered')
  ax[0,2].plot([x['epoch'] for x in audits],[x['online']['ce'] for x in audits],':',color=color,label=row+' online')
  ax[1,0].plot(xs,[x['mean_loss'] for x in epochs],color=color,label=row)
  ax[1,1].plot(xs,[x['terms']['sat_cls'] for x in epochs],color=color,label=row)
  bins=[sum(x['field_evaluations']>1 for x in actions if start<=x['epoch']<start+10) for start in range(1,201,10)]
  ax[1,2].step(np.arange(5,201,10),bins,where='mid',color=color,label=row)
 ax[0,0].set(title='Actual perturbation fraction in satellite-view batch',ylabel='Selected fraction',ylim=(0,1))
 ax[0,1].axhline(1/3,color='#555',ls='--',label='First effective scene switch')
 ax[0,1].axhline(2/3,color='#888',ls=':',label='Final-stage threshold')
 ax[0,1].set(title='4 level increments, only 1 effective scene switch',ylabel='Curriculum level',ylim=(0,1))
 ax[0,2].set(title='Recovered-head monitor CE worsens at every audit',ylabel='Cross-entropy (log scale)',yscale='log')
 ax[1,0].set(title='Full epoch loss (different training exposure)',ylabel='Mean loss')
 ax[1,1].set(title='Unweighted satellite TX CE',ylabel='CE (direct term begins E80)')
 ax[1,2].set(title='Actual extragradient corrections',ylabel='Corrections per 10 epochs')
 for panel in ax.flat:
  panel.set_xlabel('Epoch');panel.legend(fontsize=8,loc='best')
 fig.suptitle('C2 vs S4 | complete E1-E200 evidence | seed 392005',fontsize=16,fontweight='bold')
 fig.savefig(out/'core90_c2_s4_mechanisms.png',dpi=180);plt.close(fig)
 scenes=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak'];rxs=['0','2','5','7','9','10','11']
 delta=np.array([[100*(summary['C2']['target_scores']['scenes'][s]['per_rx'][r]['accuracy']-summary['S4']['target_scores']['scenes'][s]['per_rx'][r]['accuracy']) for r in rxs] for s in scenes])
 fig,ax=plt.subplots(figsize=(10.5,4.4),layout='constrained')
 im=ax.imshow(delta,cmap='RdBu',vmin=-9,vmax=9,aspect='auto');ax.grid(False)
 ax.set_xticks(range(7),['RX'+r for r in rxs]);ax.set_yticks(range(4),['Clean','LEO clear','LEO low elevation','LEO rain'])
 for y in range(4):
  for x in range(7):ax.text(x,y,f'{delta[y,x]:+.2f}',ha='center',va='center',color='white' if abs(delta[y,x])>5 else '#202020',fontsize=12)
 ax.set_title('Target accuracy: C2 minus S4 (percentage points)',fontsize=15,pad=12)
 fig.colorbar(im,ax=ax,label='Positive: C2 higher; negative: S4 higher')
 fig.savefig(out/'core90_c2_s4_target_delta.png',dpi=180);plt.close(fig)
 print('Rendered 2 evidence figures')

if __name__=='__main__':main()
