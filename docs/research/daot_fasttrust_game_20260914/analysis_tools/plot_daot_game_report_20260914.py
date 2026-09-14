from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path('E:/type10-7/automation_reports/CV-SincNet/daot_fasttrust_game_comprehensive_20260914/report_bundle')
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
colors=['#202d46','#e18b3c','#3f8598','#9584bc','#bd5660','#688749'];ids=['F-A1','F-M11','F-M08','F-M05','F-M14','F-M12']
fig,ax=plt.subplots(2,2,figsize=(14,10));fig.subplots_adjust(hspace=.36,wspace=.25)
scores={r['row']:r for r in rows(B/'sources/XUC_FULL/test_summary.csv')}
ax[0,0].scatter(ids,[float(scores[r]['leo_mean'])for r in ids],c=colors,s=95);ax[0,0].set_ylim(64,70);ax[0,0].set_ylabel('LEO mean accuracy (%)');ax[0,0].set_title('A. Fixed final E200, one seed')
for i,r in enumerate(ids):ax[0,0].text(i,float(scores[r]['leo_mean'])+.12,f"{float(scores[r]['leo_mean']):.3f}",ha='center',fontsize=9)
summary=json.loads((B/'tables/summary.json').read_text(encoding='utf-8'));ss={r['row']:r for r in summary['full_summary']};bot=[0]*6
for k,c,label in [('hard_count_sum','#3f8598','H'),('partial_count_sum','#dca14d','P'),('negative_count_sum','#bd5660','N')]:
 vals=[ss[r][k]/1e6 for r in ids];ax[0,1].bar(ids,vals,bottom=bot,color=c,label=label);bot=[a+b for a,b in zip(bot,vals)]
ax[0,1].set_ylabel('Selection events (millions; not unique IQ)');ax[0,1].set_title('B. RC4 routing changes with the training path');ax[0,1].legend(frameon=False)
curves=rows(B/'sources/XUC_FULL/training_curves.csv')
for rid,c in zip(ids,colors):
 rr=[r for r in curves if r['row']==rid];v=[float(r['source_val_accuracy']) for r in rr]
 if v and max(v)<=1:v=[100*x for x in v]
 ax[1,0].plot([int(r['epoch'])for r in rr],v,label=rid,color=c,lw=1.1)
ax[1,0].set_ylim(94,100);ax[1,0].set_xlim(20,200);ax[1,0].set_xlabel('Epoch');ax[1,0].set_ylabel('Source V accuracy (%)');ax[1,0].set_title('C. Similar source V does not imply equal target accuracy');ax[1,0].legend(ncol=3,fontsize=8,frameon=False)
aud=rows(B/'tables/full_audits.csv')
for rid,c in [('F-M11',colors[1]),('F-M08',colors[2])]:
 rr=[r for r in aud if r['row']==rid and r['kind']=='legacy'];ax[1,1].plot([int(r['step'])for r in rr],[float(r['gap'])for r in rr],label=rid,color=c,lw=1)
ax[1,1].axhline(0,color='black',ls='--',lw=.8);ax[1,1].set_xlabel('Accepted main step');ax[1,1].set_ylabel('CE online - CE recovered');ax[1,1].set_title('D. All observed legacy recoveries worsen monitor CE');ax[1,1].legend(frameon=False)
for a in ax.flat:a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.16);a.set_axisbelow(True)
(B/'figures').mkdir(exist_ok=True);fig.savefig(B/'figures/fusion_diagnostics.png',dpi=170,bbox_inches='tight');fig.savefig(B/'figures/fusion_diagnostics.svg',bbox_inches='tight');print('saved')
