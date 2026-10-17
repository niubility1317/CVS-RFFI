"""Scientific figures from independent complete source-stat aggregates only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def draw(root):
 root=Path(root);s=json.loads((root/'analysis_summary.json').read_text());prefix=json.loads((root/'complete_measurement_prefix.json').read_text())
 if s['execution']!='VERIFIED' or s['count']!=5040 or s['target_access'] or s['new_NN']:raise ValueError('Complete source analysis first')
 out=root/'figures';out.mkdir(exist_ok=False);plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False});paths=[]
 def save(fig,name):
  fig.tight_layout()
  for ext in ('png','pdf'):
   p=out/(name+'.'+ext);fig.savefig(p,dpi=170,bbox_inches='tight');paths.append(p.name)
  plt.close(fig)
 fig,ax=plt.subplots(figsize=(14,5));cols=[(r,dd) for r in (3,4,6,8) for dd in (1,2,3)];arr=np.array([[100*next(c['fraction'] for c in s['cells'] if (c['TX'],c['RX'],c['day'])==(t,r,dd)) for r,dd in cols] for t in range(6)]);im=ax.imshow(arr,vmin=0,vmax=100,cmap='Blues',aspect='auto');ax.set_xticks(range(12),['RX'+str(r)+'\nday'+str(dd) for r,dd in cols]);ax.set_yticks(range(6),['TX'+str(t) for t in range(6)]);ax.set_title('Observable affine-grid coverage (%); all 72 source cells, n=70 each')
 for i in range(6):
  for j in range(12):ax.text(j,i,f'{arr[i,j]:.1f}',ha='center',va='center',color='white' if arr[i,j]>65 else 'black')
 fig.colorbar(im,ax=ax,label='Percent; preregistered requirement >=95 in every cell');save(fig,'all_source_cell_observability')
 fig,axes=plt.subplots(1,2,figsize=(13,4.5));names=list(s['reasons']);counts=[s['reasons'][n] for n in names];axes[0].bar(range(len(names)),counts,color='#4477AA');axes[0].set_xticks(range(len(names)),[n.replace('_','\n') for n in names]);axes[0].set_ylabel('Source packets (all 5040 retained)');axes[0].set_title('Recorded measurement outcomes')
 for i,n in enumerate(counts):axes[0].text(i,n+max(counts)*.02,str(n),ha='center')
 axes[0].set_ylim(0,max(counts)*1.14);axes[1].plot([p['processed'] for p in prefix],[100*p['fraction'] for p in prefix],color='#009E73');axes[1].axhline(95,color='#D55E00',linestyle='--',label='Coverage criterion');axes[1].set_ylim(-2,102);axes[1].set_xlabel('Complete source measurement prefix (packet order)');axes[1].set_ylabel('Observable fraction (%)');axes[1].set_title('Full source trace; no training occurred');axes[1].legend();fig.suptitle('Processed eq1/RMS IQ; no ADC-causality or identity-performance claim',y=1.02);save(fig,'measurement_outcomes_and_complete_prefix')
 (out/'manifest.json').write_text(json.dumps(dict(status='VERIFIED',source_only=True,target_access=False,new_NN=0,files=paths),indent=2)+'\n',encoding='utf8');return [out/p for p in paths]
