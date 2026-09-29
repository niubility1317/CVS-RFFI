import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
base=Path('E:/type10-7/code/worktrees/ir_eg_v1_20260927/automation_reports/CV-SincNet/20260928-phase1-ir-eg-source-screen-s392005-r01/analysis_20260929')
rows=list(csv.DictReader((base/'epoch_metrics.csv').open(encoding='utf-8')))
names=list(dict.fromkeys(r['row'] for r in rows));colors=['#176b91','#e78324','#27946c','#8a68b0','#bf5576','#6d7278']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white'})
fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for name,c in zip(names,colors):
 r=[x for x in rows if x['row']==name];x=np.array([int(z['epoch']) for z in r]);label=name.replace('_s392005','')
 def vals(k):return np.array([float(z[k]) if z[k] else np.nan for z in r])
 axs[0,0].plot(x,vals('source_accuracy')*100,color=c,label=label,lw=1.3)
 axs[0,1].plot(x,vals('mean_loss'),color=c,lw=1.3)
 axs[0,2].plot(x,vals('elapsed_seconds')/3600,color=c,lw=1.5)
 axs[1,0].plot(x,vals('daot_rc4.partial_count'),color=c,lw=1.3)
 axs[1,1].semilogy(x,vals('daot_rc4.weighted_identity')+1e-12,color=c,lw=1.2)
 axs[1,2].plot(x,np.convolve(vals('epoch_seconds'),np.ones(5)/5,mode='valid').tolist()+[np.nan]*4,color=c,lw=1.3)
titles=['Full source V accuracy','Total training loss (changing objective)','Cumulative wall time','P-route samples per U batch','Weighted RC4 identity loss','Epoch wall time (5-epoch mean)']
ylabs=['Accuracy (%)','Loss','Hours','Mean count','Mean weighted loss','Seconds']
for ax,title,yl in zip(axs.flat,titles,ylabs):
 ax.set(title=title,xlabel='Epoch',ylabel=yl,xlim=(1,200));ax.grid(alpha=.17)
 for boundary in [21,80,91,161]:ax.axvline(boundary,color='gray',ls=':',alpha=.25,lw=.8)
axs[0,0].set_ylim(95,99.1)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=6,frameon=False)
fig.suptitle('IR-EG audit | seed 392005 | full logs through power loss\nSource accuracy panel zooms to 95–99.1%; IR variants end at E163/164',fontsize=14)
fig.savefig(base/'training_diagnosis.png',dpi=150)
fig.savefig(base/'training_diagnosis.pdf')
print(base/'training_diagnosis.png')
