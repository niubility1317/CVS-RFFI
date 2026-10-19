import json,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
args=argparse.ArgumentParser();args.add_argument('--input',required=True);args.add_argument('--output',required=True);a=args.parse_args();p=Path(a.input);x=json.loads(p.read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':180})
colors=['#295A83','#C06C37','#4D8A73'];names=['PAMF','CSIL','MoPC-HR'];keys=['PAMF','csil','mopc_hr'];fig,axs=plt.subplots(1,2,figsize=(11,4.3))
for i,k in enumerate(keys):
 s=x['same_observation'][k];vals=[100*s['clean']['accuracy'],100*s['satellite']['practical_mid_urban']['accuracy'],100*s['satellite']['practical_low_urban']['accuracy']];pos=np.arange(3)+(i-1)*.24;axs[0].bar(pos,vals,.22,color=colors[i],label=names[i])
 for a,b in zip(pos,vals):axs[0].text(a,b+.8,f'{b:.1f}',ha='center',fontsize=8)
axs[0].set_xticks(range(3),['Clean','Mid urban','Low urban']);axs[0].set_ylim(0,107);axs[0].set_ylabel('Accuracy (%)');axs[0].legend(frameon=False,loc='lower left');axs[0].set_title('Same source observations, frozen native heads',loc='left',fontsize=11)
for i,k in enumerate(keys):
 vals=[]
 for dim,st in [('state','BLOCKED'),('signal_to_added_noise_db','<3')]:
  r=next(r for r in x['strata'] if r['method']==k and r['scene']=='practical_low_urban' and r['dimension']==dim and r['stratum']==st);vals.append(100*r['accuracy'])
 pos=np.arange(2)+(i-1)*.24;axs[1].bar(pos,vals,.22,color=colors[i])
 for a,b in zip(pos,vals):axs[1].text(a,b+.8,f'{b:.1f}',ha='center',fontsize=8)
axs[1].set_xticks(range(2),['BLOCKED (n=1,979)','Added-noise ratio <3 dB (n=1,127)']);axs[1].set_ylim(0,85);axs[1].set_ylabel('Low urban accuracy (%)');axs[1].set_title('Advantage persists in damaged conditions',loc='left',fontsize=11)
fig.text(.07,.01,'V: 1,800 physical samples; 2 urban channel draws. One model per method; E30 vs E200 training budgets differ.',fontsize=8,color='#555555');fig.tight_layout(rect=(0,.045,1,1));fig.savefig(out/'source_model_comparison.png',bbox_inches='tight');fig.savefig(out/'source_model_comparison.pdf',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,3.6));
for i,k in enumerate(('L','V')):
 vals=[100*x['fresh_L_V'][k][s]['accuracy'] for s in ('clean','practical_mid_urban','practical_low_urban')];pos=np.arange(3)+(i-.5)*.28;ax.bar(pos,vals,.26,label=k,color=('#295A83','#C06C37')[i]);
 for a,b in zip(pos,vals):ax.text(a,b+1,f'{b:.2f}',ha='center',fontsize=9)
ax.set_xticks(range(3),['Clean','Fresh mid urban','Fresh low urban']);ax.set_ylim(0,108);ax.set_ylabel('Frozen PAMF accuracy (%)');ax.legend(frameon=False);ax.set_title('Seen source physical samples show a small fresh-channel gap',loc='left',fontsize=11);fig.text(.1,.01,'Balanced L/V: 1,800 physical samples each; 2 new channel draws. No fitting or state update.',fontsize=8,color='#555555');fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'fresh_L_V_comparison.png',bbox_inches='tight');fig.savefig(out/'fresh_L_V_comparison.pdf',bbox_inches='tight');print(json.dumps({'plots':4}))
