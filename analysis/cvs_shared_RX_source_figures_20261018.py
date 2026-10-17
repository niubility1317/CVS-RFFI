"""Plot saved aggregate source statistics; no IQ/checkpoint or target read."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'automation_reports/CV-SincNet/20261018-diagnostic-F-sharedRX-sourceLtoV-manysig-m1-r01'
r=json.loads((REPORT/'source_result.json').read_text(encoding='utf-8'))
rx=[1,3,4,6,8]
colors=['#355c7d','#c06c84','#417d6d','#b98928','#7b5f9b']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(1,2,figsize=(10.7,4.35),gridspec_kw={'width_ratios':[1.3,1]})
for i,(receiver,color) in enumerate(zip(rx,colors)):
 x=[f['prediction'][i] for f in r['folds']]
 y=[f['V_RX_contrast'][i] for f in r['folds']]
 ax[0].scatter(x,y,s=28,c=color,label=f'RX{receiver}',edgecolors='white',linewidths=.35,alpha=.85)
lo,hi=-.085,.145
ax[0].plot([lo,hi],[lo,hi],color='#6b7280',ls='--',lw=1)
ax[0].axhline(0,color='#d1d5db',lw=.7);ax[0].axvline(0,color='#d1d5db',lw=.7)
ax[0].set(xlim=(lo,hi),ylim=(lo,hi),xlabel='L-only predicted RX contrast',ylabel='Observed V RX contrast',title='A  All 90 source V cells')
ax[0].legend(ncol=3,frameon=False,fontsize=8,loc='upper left')
gains=[100*r['per_RX'][str(i)]['risk_gain'] for i in rx]
ax[1].bar(range(5),gains,color=colors,width=.67)
ax[1].set(xticks=range(5),xticklabels=[f'RX{i}' for i in rx],ylim=(0,105),ylabel='Relative SSE reduction (%)',title='B  Each source RX improves')
ax[1].axhline(100*r['risk_gain'],color='#111827',ls='--',lw=1,label=f"Pooled: {100*r['risk_gain']:.2f}%")
for i,v in enumerate(gains):ax[1].text(i,v+1.4,f'{v:.2f}',ha='center',fontsize=9)
ax[1].legend(frameon=False,fontsize=9,loc='upper right')
fig.suptitle('L6300 to V27000: fixed scalar association, all five source RX',y=1.015,fontsize=12)
fig.text(.02,-.035,'18 overlapping TX/day analytical folds; no held training RX. Scalar prediction only; no IQ augmentation or recognition gain.',fontsize=9,color='#374151')
fig.tight_layout(w_pad=2.5)
out=REPORT/'figures';out.mkdir(exist_ok=True)
for suffix in ('png','pdf'):fig.savefig(out/('source_L_to_V.'+suffix),dpi=180,bbox_inches='tight')
plt.close(fig)
print(json.dumps({'status':'PLOT_WRITTEN','RX':5,'V_cells':90,'input':'saved aggregate source_result.json','IQ':0,'NN':0,'target':False}))
