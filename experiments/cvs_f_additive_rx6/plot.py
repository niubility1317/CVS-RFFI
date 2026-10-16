"""Scientific source-only charts from independently recomputed scalars."""
import argparse,csv,json,math
from pathlib import Path
ARMS=('D00','D_raw','D_As');VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
SHORT=('Clean','High suburban','Mid suburban','Low suburban','High urban','Mid urban','Low urban')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def draw(analysis,output):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 analysis=Path(analysis);output=Path(output);output.mkdir(parents=True,exist_ok=False)
 r=read(analysis/'analysis_summary.json');m=read(analysis/'mechanism_analysis.json')
 with (analysis/'training_curves.csv').open(encoding='utf8',newline='') as f:curves=list(csv.DictReader(f))
 with (analysis/'source_strata.csv').open(encoding='utf8',newline='') as f:tx=list(csv.DictReader(f))
 colors={'D00':'#33658A','D_raw':'#5C946E','D_As':'#D56A32'};names=[]
 def save(fig,name):
  fig.suptitle('Held source RX6 | seed 2026092701 | scratch E20 | default cosine head',fontsize=10)
  for ext in ('png','pdf'):fig.savefig(output/(name+'.'+ext),dpi=180)
  plt.close(fig);names.append(name+'.png')
 fig,axes=plt.subplots(1,2,figsize=(13.5,4.6),constrained_layout=True)
 for j,a in enumerate(ARMS):
  val=[100*r['scores'][a][v]['accuracy'] for v in VIEWS];bars=axes[0].bar(np.arange(7)+(j-1)*.25,val,width=.24,color=colors[a],label=a);axes[0].bar_label(bars,fmt='%.1f',padding=2,fontsize=6.8)
 axes[0].set_xticks(range(7),SHORT,rotation=35,ha='right');axes[0].set_ylim(0,100);axes[0].set_ylabel('Accuracy (%)');axes[0].legend(fontsize=8)
 for j,ref in enumerate(ARMS[:2]):
  vals=[r['paired'][ref][v]['accuracy_gain_pp'] for v in VIEWS];bars=axes[1].bar(np.arange(7)+(j-.5)*.34,vals,width=.32,color=colors[ref],label='D_As - '+ref);axes[1].bar_label(bars,fmt='%+.1f',padding=3,fontsize=7.5)
 axes[1].axhline(0,color='grey',lw=.8);axes[1].set_xticks(range(7),SHORT,rotation=35,ha='right');axes[1].set_ylabel('Paired gain (percentage points)');axes[1].legend(fontsize=8);save(fig,'source_accuracy')
 fig,axes=plt.subplots(2,2,figsize=(11.5,7.5),constrained_layout=True)
 for a in ARMS:
  rows=[x for x in curves if x['arm']==a];e=[int(x['epoch']) for x in rows]
  for ax,key,title in zip(axes.flat,('base_total_L','extra_pair_loss','val_tx_acc','supcon'),('Common full L objective','Additional pair CE + RDrop KL','Seen-source V accuracy (%)','Ordinary-view SupCon')):
   yy=[float(x[key]) for x in rows];ax.plot(e,yy,color=colors[a],label=a);ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
 for ax in axes.flat:ax.axvline(13.5,color='grey',ls=':',lw=1);ax.legend(fontsize=8);ax.set_xticks([1,5,10,13,15,20])
 save(fig,'training_curves')
 fig,axes=plt.subplots(1,3,figsize=(14,4.6),constrained_layout=True)
 dimensions=('state','lock','quality_snr_db');orders=(['LOS','PARTIAL','BLOCKED'],['locked','degraded','unlocked'],['-inf:5','5:15','15:inf'])
 for ax,dim,order in zip(axes,dimensions,orders):
  cells=[x for x in m['strata'] if x['dimension']==dim and x['view'] in VIEWS[4:]];labels=[];values=[];counts=[]
  for view in VIEWS[4:]:
   for value in order:
    c=next((x for x in cells if x['view']==view and x['value']==value),None)
    if c is None:continue
    labels.append(SHORT[VIEWS.index(view)].replace(' urban','')+' / '+value);values.append(c['paired']['D00']['accuracy_gain_pp']);counts.append(c['count'])
  pos=np.arange(len(labels));ax.barh(pos,values,color=[colors['D_As'] if v>=0 else '#A34B62' for v in values]);ax.axvline(0,color='grey',lw=.8);ax.set_yticks(pos,[v+' (n='+str(n)+')' for v,n in zip(labels,counts)],fontsize=7);ax.invert_yaxis();ax.set_xlabel('D_As - D00 (pp)');ax.set_title(dim)
 save(fig,'urban_strata')
 fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
 for a in ARMS:
  s=m['margins_and_retention'][a];axes[0].plot(range(7),[100*s[v]['fixed_D00_clean_correct_retention'] for v in VIEWS],marker='o',color=colors[a],label=a);axes[1].plot(range(7),[s[v]['fixed_margin_mean'] for v in VIEWS],marker='o',color=colors[a],label=a)
 for ax in axes:ax.set_xticks(range(7),SHORT,rotation=35,ha='right');ax.grid(alpha=.2);ax.legend(fontsize=8)
 axes[0].set_ylim(0,101);axes[0].set_ylabel('Accuracy on fixed D00-clean-correct set (%)');axes[0].set_title('Fixed source packet set, n='+str(m['fixed_D00_clean_correct_count']))
 axes[1].set_ylabel('(True - best-other logit) / 30');axes[1].axhline(0,color='grey',ls='--',lw=.8);axes[1].set_title('Default-head margin on the same fixed set');save(fig,'fixed_clean_retention')
 fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
 for ax,dim,values in zip(axes,('TX','day'),(list(range(6)),[1,2,3])):
  array=np.array([[float(next(x for x in tx if x['reference']=='D00' and x['view']==v and x['dimension']==dim and int(x['value'])==t)['gain_pp']) for v in VIEWS] for t in values]);bound=max(1,float(abs(array).max()));im=ax.imshow(array,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto');ax.set_xticks(range(7),SHORT,rotation=35,ha='right');ax.set_yticks(range(len(values)),[dim+' '+str(v) for v in values]);ax.set_title('D_As - D00: all '+dim+' groups')
  for i in range(len(values)):
   for j in range(7):ax.text(j,i,f'{array[i,j]:+.1f}',ha='center',va='center',fontsize=8,color='black' if abs(array[i,j])<.65*bound else 'white')
  fig.colorbar(im,ax=ax,label='Percentage points',shrink=.8)
 save(fig,'error_redistribution')
 coverage=r['execution']['D_As']['coverage'];cells=coverage['cells'];fits=(1,3,4,8);fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
 arr=np.array([[sum(x['count'] for x in cells if x['TX']==t and x['RX']==rx) for rx in fits] for t in range(6)]);im=axes[0].imshow(arr,cmap='Blues',aspect='auto');axes[0].set_xticks(range(4),['RX'+str(v) for v in fits]);axes[0].set_yticks(range(6),['TX'+str(v) for v in range(6)]);axes[0].set_title('All labeled satellite observations by TX / fit RX')
 for i in range(6):
  for j in range(4):axes[0].text(j,i,str(arr[i,j]),ha='center',va='center',fontsize=8)
 fig.colorbar(im,ax=axes[0],label='Packets')
 states=['LOS','PARTIAL','BLOCKED'];bottom=np.zeros(6)
 for state,color in zip(states,('#5B8E7D','#D4A373','#B35C5C')):
  vals=np.array([sum(x['count'] for x in cells if x['scene']==v and x['state']==state) for v in VIEWS[1:]]);axes[1].bar(range(6),vals,bottom=bottom,color=color,label=state);bottom+=vals
 axes[1].set_xticks(range(6),SHORT[1:],rotation=35,ha='right');axes[1].set_ylabel('Training observations');axes[1].set_title('Full nominal prior, total='+str(coverage['total']));axes[1].legend(fontsize=8);save(fig,'training_coverage')
 return names
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(draw(a.analysis,a.output)))
