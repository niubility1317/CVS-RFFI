"""Static source-only figures from independently recomputed outputs."""
import argparse,csv,json
from pathlib import Path
from .analyze import ARMS,NEW,VIEWS,read
SHORT=('Clean','High suburban','Mid suburban','Low suburban','High urban','Mid urban','Low urban')
def draw(root,output):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False);r=read(root/'analysis_summary.json');m=read(root/'mechanism_analysis.json')
 with (root/'training_curves.csv').open(encoding='utf8',newline='') as f:curves=list(csv.DictReader(f))
 with (root/'source_strata.csv').open(encoding='utf8',newline='') as f:tx=list(csv.DictReader(f))
 colors={'D_raw':'#4A7A5C','SAME_RX':'#3D6D9B','CROSS_RX':'#C86A38'};names=[]
 def save(fig,name):
  fig.suptitle('Held source RX1 | seed '+str(r['model_seed'])+' | scratch E20 | default cosine head',fontsize=10)
  for ext in ('png','pdf'):fig.savefig(output/(name+'.'+ext),dpi=180)
  plt.close(fig);names.append(name+'.png')
 fig,axes=plt.subplots(1,2,figsize=(13.5,4.8),constrained_layout=True)
 for j,a in enumerate(ARMS):
  bars=axes[0].bar(np.arange(7)+(j-1)*.25,[100*r['scores'][a][v]['accuracy'] for v in VIEWS],width=.24,color=colors[a],label=a);axes[0].bar_label(bars,fmt='%.1f',padding=3,fontsize=7,rotation=90)
 for j,ref in enumerate(ARMS[:2]):
  bars=axes[1].bar(np.arange(7)+(j-.5)*.34,[r['paired'][ref][v]['accuracy_gain_pp'] for v in VIEWS],width=.32,color=colors[ref],label='CROSS - '+ref);axes[1].bar_label(bars,fmt='%+.1f',padding=3,fontsize=8)
 for ax in axes:ax.set_xticks(range(7),SHORT,rotation=35,ha='right');ax.legend(fontsize=8);ax.axhline(0,color='grey',lw=.6)
 axes[0].set_ylim(0,100);axes[0].set_ylabel('Accuracy (%)');axes[1].set_ylabel('Paired gain (percentage points)');save(fig,'source_accuracy')
 fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
 for a in ARMS:
  rows=[x for x in curves if x['arm']==a];e=[int(x['epoch']) for x in rows]
  for ax,key,title in zip(axes.flat,('base_total_L','extra_pair_loss','val_tx_acc','raw_CE','ordinary_CE','supcon'),('Common L objective','Extra pair CE + RDrop KL','Seen-source V accuracy (%)','Common raw CE','Common ordinary CE','Ordinary-view SupCon')):
   ax.plot(e,[float(x[key]) for x in rows],color=colors[a],label=a);ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.axvline(13.5,color='grey',ls=':',lw=.7)
 for ax in axes.flat:ax.legend(fontsize=7);ax.set_xticks([1,5,10,13,15,20])
 save(fig,'training_curves')
 fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
 for ax,dim,order in zip(axes,('state','lock','quality_snr_db'),(['LOS','PARTIAL','BLOCKED'],['locked','degraded','unlocked'],['-inf:5','5:15','15:inf'])):
  cells=[x for x in m['strata'] if x['dimension']==dim and x['view'] in VIEWS[4:]];labels=[];vals=[]
  for view in VIEWS[4:]:
   for value in order:
    c=next((x for x in cells if x['view']==view and x['value']==value),None)
    if c is None:continue
    labels.append(SHORT[VIEWS.index(view)].replace(' urban','')+' / '+value+' (n='+str(c['count'])+')');vals.append(c['paired']['D_raw']['accuracy_gain_pp'])
  pos=np.arange(len(vals));ax.barh(pos,vals,color=[colors['CROSS_RX'] if x>=0 else '#A84966' for x in vals]);ax.axvline(0,color='grey',lw=.7);ax.set_yticks(pos,labels,fontsize=7);ax.invert_yaxis();ax.set_title(dim);ax.set_xlabel('CROSS_RX - D_raw (pp)')
 save(fig,'urban_strata')
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
 for a in ARMS:
  s=m['margins_and_retention'][a];axes[0].plot(range(7),[100*s[v]['fixed_raw_clean_correct_retention'] for v in VIEWS],marker='o',color=colors[a],label=a)
  if a in NEW:axes[1].plot(range(7),[s[v]['scaled_margin_fixed']['mean'] for v in VIEWS],marker='o',color=colors[a],label=a)
 for ax in axes:ax.set_xticks(range(7),SHORT,rotation=35,ha='right');ax.grid(alpha=.2);ax.legend(fontsize=8)
 axes[0].set_ylabel('Accuracy on fixed raw-clean-correct set (%)');axes[0].set_title('Fixed packet set, n='+str(m['fixed_raw_clean_correct_count']));axes[0].set_ylim(0,101);axes[1].set_title('Raw reference per-packet margin unavailable');axes[1].set_ylabel('(True - best other logit) / 30');axes[1].axhline(0,color='grey',ls=':',lw=.8);save(fig,'fixed_clean_retention')
 fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
 for ax,dim,values in zip(axes,('TX','day'),(list(range(6)),[1,2,3])):
  arr=np.array([[float(next(x for x in tx if x['reference']=='D_raw' and x['view']==v and x['dimension']==dim and int(x['value'])==t)['gain_pp']) for v in VIEWS] for t in values]);bound=max(1,float(abs(arr).max()));im=ax.imshow(arr,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto');ax.set_xticks(range(7),SHORT,rotation=35,ha='right');ax.set_yticks(range(len(values)),[dim+' '+str(v) for v in values]);ax.set_title('CROSS_RX - D_raw: all '+dim+' groups')
  for i in range(len(values)):
   for j in range(7):ax.text(j,i,f'{arr[i,j]:+.1f}',ha='center',va='center',fontsize=8,color='black' if abs(arr[i,j])<.65*bound else 'white')
  fig.colorbar(im,ax=ax,label='Percentage points',shrink=.8)
 save(fig,'error_redistribution')
 fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
 for a in NEW:
  cells=r['mix_audits'][a]['relative_L2_histogram'];axes[0].step([x['upper'] for x in cells],[100*x['count']/568320 for x in cells],where='pre',color=colors[a],label=a)
 axes[0].set_xlabel('Relative IQ L2 change');axes[0].set_ylabel('Packets (%)');axes[0].set_title('Actual IQ interpolation, all 568320 / row');axes[0].legend(fontsize=8)
 fits=(3,4,6,8)
 for ax,a in zip(axes[1:],NEW):
  cells=r['mix_audits'][a]['coverage'];arr=np.array([[sum(c['count'] for c in cells if c['primary_RX']==rx and c['partner_RX']==other) for other in fits] for rx in fits]);im=ax.imshow(arr,cmap='Blues',aspect='auto',vmin=0,vmax=max(1,float(arr.max()))*1.3);ax.set_xticks(range(4),['RX'+str(x) for x in fits]);ax.set_yticks(range(4),['RX'+str(x) for x in fits]);ax.set_xlabel('Donor');ax.set_ylabel('Anchor');ax.set_title(a+' partner coverage')
  for i in range(4):
   for j in range(4):ax.text(j,i,str(arr[i,j]),ha='center',va='center',fontsize=8)
 save(fig,'IQ_mixing_and_coverage')
 if (root/'crossseed_analysis.json').is_file():
  cross=read(root/'crossseed_analysis.json');rows=cross['rows'];fig,axes=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True)
  for ax,v,title in zip(axes[0],('clean','practical_low_urban'),('Clean identity accuracy','Low-urban accuracy')):
   for j,a in enumerate(ARMS):
    bars=ax.bar(np.arange(2)+(j-1)*.25,[100*x['scores'][a][v]['accuracy'] for x in rows],width=.24,color=colors[a],label=a);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8,rotation=90)
   ax.set_ylim(0,100);ax.set_ylabel('Accuracy (%)');ax.set_title(title);ax.legend(fontsize=7)
  for j,ref in enumerate(ARMS[:2]):
   vals=[x['qualification']['comparisons'][ref]['all_view_gain_pp']['clean'] for x in rows];bars=axes[1,0].bar(np.arange(2)+(j-.5)*.3,vals,width=.28,color=colors[ref],label='CROSS - '+ref);axes[1,0].bar_label(bars,fmt='%+.2f',padding=3,fontsize=8)
  for j,(key,title) in enumerate((('urban','Urban mean'),('low','Low urban'))):
   vals=[x['qualification']['comparisons']['D_raw']['urban_mean_gain_pp'] if key=='urban' else x['qualification']['comparisons']['D_raw']['all_view_gain_pp']['practical_low_urban'] for x in rows];bars=axes[1,1].bar(np.arange(2)+(j-.5)*.3,vals,width=.28,color='#A26A3A' if j==0 else '#D7A56B',label=title);axes[1,1].bar_label(bars,fmt='%+.2f',padding=3,fontsize=8)
  axes[1,0].set_title('Paired clean gain');axes[1,1].set_title('CROSS_RX - D_raw practical gain')
  for ax in axes.flat:ax.set_xticks(range(2),[str(x['model_seed']) for x in rows])
  for ax in axes[1]:ax.axhline(0,color='grey',lw=.7);ax.set_ylabel('Percentage points');ax.legend(fontsize=7)
  fig.suptitle('Same held source RX1 and physical channel draws | two scratch E20 seeds | default head',fontsize=10)
  for ext in ('png','pdf'):fig.savefig(output/('crossseed_comparison.'+ext),dpi=180)
  plt.close(fig);names.append('crossseed_comparison.png')
 return names
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(draw(a.input,a.output)))
