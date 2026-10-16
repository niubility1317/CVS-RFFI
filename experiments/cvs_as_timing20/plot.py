"""Finalized independent source-only plots;no IQ/checkpoint/model access."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .analyze import VIEWS,ARMS
LABELS=('Clean','High suburban','Mid suburban','Low suburban','High urban','Mid urban','Low urban')
COLORS=dict(RAW='#777777',ALL='#2878b5',EARLY7='#3a9b74',LATE7='#e07b39')
def read(p):return json.loads(Path(p).read_text())
def draw(root):
 root=Path(root);a=read(root/'analysis_summary.json');curves=read(root/'training_curves.json');strata=read(root/'source_strata.json');out=root/'figures';out.mkdir(exist_ok=False)
 if a['execution']!='VERIFIED' or len(curves)!=80 or a['target_access']:raise ValueError('Complete independent source matrix required')
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140});names=[]
 def save(fig,name):
  fig.savefig(out/(name+'.png'),bbox_inches='tight',dpi=170);fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig);names.append(name)
 fig,ax=plt.subplots(figsize=(14,5.4),constrained_layout=True)
 for j,arm in enumerate(ARMS):
  v=[100*a['scores'][arm][s]['accuracy'] for s in VIEWS];bars=ax.bar(np.arange(7)+(j-1.5)*.19,v,width=.18,label=arm,color=COLORS[arm]);ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
 ax.set_ylim(0,105);ax.set_xticks(range(7),LABELS,rotation=20,ha='right');ax.set_ylabel('Default-head accuracy (%)');ax.legend(ncol=4);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_title('Held source RX1 L1260, seed2026092701: fixed FINAL20, clean + six unchanged practical');save(fig,'source_scene_results')
 fig,axs=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
 panels=(('total_L','Total labeled objective'),('extra_CE','Extra pair CE'),('extra_KL','Extra pair RDrop KL'),('seenV_accuracy','Seen-source V accuracy (%)'),('grad_before_clip','Mean gradient before clipping'),('pseudo_selected','Old-gate selected U (epoch SUM)'))
 for ax,(key,title) in zip(axs.flat,panels):
  for arm in ARMS:
   q=[r for r in curves if r['arm']==arm];ax.plot([r['epoch'] for r in q],[r[key] for r in q],label=arm,color=COLORS[arm],lw=1.5)
  ax.axvline(7.5,color='gray',ls=':',lw=1);ax.axvline(13.5,color='gray',ls='--',lw=1);ax.set_title(title);ax.set_xlabel('Epoch');ax.set_xticks([1,5,7,10,13,14,17,20]);ax.grid(alpha=.2)
 axs[0,0].legend(ncol=4,fontsize=8);fig.suptitle('All20 epochs: EARLY As E1-7; LATE As E14-20; original SSL starts E14',fontsize=13);save(fig,'complete_training_curves')
 fig,axs=plt.subplots(1,3,figsize=(16,5.5),constrained_layout=True)
 for ax,dim,order,title in ((axs[0],'state',('LOS','PARTIAL','BLOCKED'),'Propagation state'),(axs[1],'lock',('locked','degraded','unlocked'),'Original receiver lock'),(axs[2],'quality',('<0','[0,5)','[5,10)','[10,20)','[20,30)','>=30'),'Original quality proxy (dB)')):
  groups={z['value']:z for z in strata if z['arm']=='RAW' and z['scene']==VIEWS[-1] and z['dimension']==dim};order=[v for v in order if v in groups]
  for j,arm in enumerate(ARMS):
   vals=[next(z['accuracy']*100 for z in strata if z['arm']==arm and z['scene']==VIEWS[-1] and z['dimension']==dim and z['value']==v) for v in order];ax.bar(np.arange(len(order))+(j-1.5)*.19,vals,width=.18,color=COLORS[arm],label=arm)
  ax.set_xticks(range(len(order)),[v+chr(10)+'n='+str(groups[v]['count']) for v in order],rotation=20 if dim=='quality' else 0);ax.set_ylim(0,105);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axs[0].set_ylabel('Low-urban default-head accuracy (%)');axs[0].legend(fontsize=8,ncol=2);fig.suptitle('Same held physical observations/states for every model; source-only conditional results');save(fig,'low_urban_strata')
 fig,axs=plt.subplots(1,2,figsize=(14,5.5),constrained_layout=True)
 for j,arm in enumerate(ARMS):axs[0].bar(np.arange(6)+(j-1.5)*.19,[100*a['retention_on_fixed_RAW_clean_correct'][s][arm]['accuracy'] for s in VIEWS[1:]],width=.18,color=COLORS[arm],label=arm)
 axs[0].set_xticks(range(6),LABELS[1:],rotation=25,ha='right');axs[0].set_ylim(0,105);axs[0].set_ylabel('Accuracy on fixed RAW clean-correct set (%)');axs[0].set_title('Fixed reference set n='+str(a['fixed_RAW_clean_correct_count']));axs[0].legend(ncol=2,fontsize=8)
 for j,ref in enumerate(('RAW','ALL','EARLY7')):axs[1].bar(np.arange(7)+(j-1)*.25,[next(z['net_correct'] for z in a['paired'] if z['reference']==ref and z['scene']==s) for s in VIEWS],width=.24,label='LATE7 vs '+ref,color=COLORS[ref])
 axs[1].axhline(0,color='black',lw=1);axs[1].set_xticks(range(7),LABELS,rotation=25,ha='right');axs[1].set_ylabel('Repairs - new errors (physical samples)');axs[1].set_title('Paired error transfer, same L1260');axs[1].legend(fontsize=8)
 for ax in axs:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 save(fig,'fixed_retention_and_paired_errors')
 fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
 for arm in ARMS:
  q=[z for z in curves if z['arm']==arm];axs[0,0].plot([z['epoch'] for z in q],[z['As_consumed'] for z in q],label=arm,color=COLORS[arm]);axs[0,1].plot([z['epoch'] for z in q],[z['lr'] for z in q],label=arm,color=COLORS[arm])
 axs[0,0].set_title('Actual consumed As samples per epoch');axs[0,0].set_xlabel('Epoch');axs[0,0].legend(ncol=2,fontsize=8);axs[0,1].set_title('Original cosine LR calendar (overlapping curves)');axs[0,1].set_xlabel('Epoch')
 for ax,dim,vals,title in ((axs[1,0],'state',('LOS','PARTIAL','BLOCKED'),'Consumed As propagation state'),(axs[1,1],'scene',VIEWS[1:],'Consumed As nominal scene')):
  for j,arm in enumerate(ARMS):
   info=a['training'][arm]['timing'];den=info['consumed'];ax.bar(np.arange(len(vals))+(j-1.5)*.19,[100*info['consumed_dimensions'][dim].get(v,0)/den if den else 0 for v in vals],width=.18,color=COLORS[arm],label=arm)
  ax.set_xticks(range(len(vals)),vals if dim=='state' else LABELS[1:],rotation=25 if dim=='scene' else 0,ha='right' if dim=='scene' else 'center');ax.set_ylabel('Consumed exposures (%)');ax.set_title(title)
 for ax in axs.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 fig.suptitle('All rows generate568320 full-physics shadows; RAW consumes0; EARLY7=LATE7 consume198912',fontsize=12);save(fig,'timing_dose_and_calendar')
 (out/'manifest.json').write_text(json.dumps(dict(status='CREATED',figures=names,source_only=True,NN=0,model_seed=2026092701,held_RX=1,target_access=False),indent=2)+chr(10))
 return [str(out/(n+'.png')) for n in names]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();print(json.dumps(draw(a.input)))
