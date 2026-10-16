"""Final source-only scientific figures;aggregates only,no model/IQ access."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ARMS=('RAW','REPLAY','REFRESH');COLORS={'RAW':'#587894','REPLAY':'#9467aa','REFRESH':'#d77439'}
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
LABELS=('clean','high sub','mid sub','low sub','high urban','mid urban','low urban')
def draw(root):
 root=Path(root);s=json.loads((root/'analysis_summary.json').read_text());curves=json.loads((root/'training_curves.json').read_text());strata=json.loads((root/'source_strata.json').read_text());out=root/'figures';out.mkdir(exist_ok=False)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':160,'pdf.fonttype':42})
 paths=[]
 def save(fig,name):
  fig.suptitle('Source held RX1 only | seed 2026092701 | fixed FINAL E20 | default cosine head',fontsize=12)
  fig.tight_layout(rect=(0,0,1,.95))
  for ext in ('png','pdf'):
   p=out/(name+'.'+ext);fig.savefig(p,bbox_inches='tight');paths.append(p)
  plt.close(fig)
 fig,axs=plt.subplots(2,1,figsize=(12,8),gridspec_kw={'height_ratios':[2,1]});x=np.arange(7)
 for j,a in enumerate(ARMS):
  vals=[100*s['scores'][a][v]['accuracy'] for v in VIEWS];bars=axs[0].bar(x+(j-1)*.25,vals,.25,label=a,color=COLORS[a])
  axs[0].bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
 axs[0].set(xticks=x,xticklabels=LABELS,ylabel='Accuracy (%)',ylim=(0,105),title='Three newly trained matched8L rows; all seven fixed readouts');axs[0].legend()
 for j,ref in enumerate(('RAW','REPLAY')):
  gain=[s['comparisons'][ref]['all_view_gain_pp'][v] for v in VIEWS];bars=axs[1].bar(x+(j-.5)*.36,gain,.36,label='REFRESH - '+ref,color=COLORS[ref]);axs[1].bar_label(bars,fmt='%+.2f',padding=3,fontsize=8)
 axs[1].axhline(0,color='black',lw=.8);axs[1].set(xticks=x,xticklabels=LABELS,ylabel='Accuracy difference (pp)',title='Source qualification: '+str(s['qualified'])+'; no formal target or Ag+As evidence');axs[1].legend();save(fig,'source_scene_results')
 fig,axs=plt.subplots(3,2,figsize=(13,11));panels=(('total_L','Complete labeled objective'),('extra_CE','Fourth IQ pair CE'),('seenV_accuracy','Seen-RX V accuracy (%)'),('supcon','Original SupCon'),('grad_before_clip','Gradient norm before clipping'),('pseudo_selected','Original U accepted per epoch'))
 for ax,(k,title) in zip(axs.flat,panels):
  for a in ARMS:
   q=sorted((z for z in curves if z['arm']==a),key=lambda z:z['epoch']);ax.plot([z['epoch'] for z in q],[z[k] for z in q],marker='.',color=COLORS[a],label=a)
  ax.axvline(13.5,color='#999999',ls=':',lw=1);ax.set(title=title,xlabel='Epoch',xticks=(1,5,10,13,15,20));ax.grid(alpha=.2)
 axs[0,0].legend();save(fig,'complete_training_curves')
 fig,axs=plt.subplots(2,2,figsize=(13,9));h=s['training']['REFRESH']['refresh'];keys=list(h['eligible_packets']);short=[k.replace('p_','') for k in keys];yy=np.arange(len(keys))
 for j,key in enumerate(('first_operator_packets','fresh_operator_packets')):
  rate=[100*h[key][k]/h['eligible_packets'][k] for k in keys];axs[0,0].barh(yy+(j-.5)*.36,rate,.36,label=('Original prePA','Fresh prePA')[j],color=('#587894','#d77439')[j])
 rate=[100*h['view_gate_disagreement_packets'][k]/h['eligible_packets'][k] for k in keys];axs[0,1].barh(yy,rate,.6,color='#9467aa')
 for ax in axs[0]:ax.set(yticks=yy,yticklabels=short);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
 axs[0,0].set(xlabel='Triggered L packets (%)',title='REFRESH native marginal gates');axs[0,0].legend();axs[0,1].set(xlabel='Original/fresh gate disagreement (%)',title='Independent observations; same native probabilities')
 q=sorted((z for z in curves if z['arm']=='REFRESH'),key=lambda z:z['epoch'])
 for key,label in [('first_prepa_raw_MSE','First - raw'),('fresh_prepa_raw_MSE','Fresh - raw'),('fresh_first_prepa_MSE','Fresh - first'),('normal_prepa_MSE','Original normal - prePA')]:axs[1,0].plot([z['epoch'] for z in q],[z[key] for z in q],label=label)
 for a in ARMS:
  q=sorted((z for z in curves if z['arm']==a),key=lambda z:z['epoch']);axs[1,1].plot([z['epoch'] for z in q],[z['selected_raw_MSE'] for z in q],label=a,color=COLORS[a])
 axs[1,0].set(xlabel='Epoch',ylabel='IQ MSE',title='Complete saved scalar records; no IQ reconstruction');axs[1,0].legend(fontsize=8);axs[1,1].set(xlabel='Epoch',ylabel='IQ MSE',title='Actual fourth-pair input versus raw');axs[1,1].legend();save(fig,'ordinary_observation_exposure')
 fig,axs=plt.subplots(2,3,figsize=(15,9))
 for col,dim in enumerate(('state','lock','quality')):
  q=[z for z in strata if z['scene']=='practical_low_urban' and z['dimension']==dim];order={'state':('LOS','PARTIAL','BLOCKED'),'lock':('locked','degraded','unlocked'),'quality':('<0','[0,5)','[5,10)','[10,20)','[20,30)','>=30')};values=[v for v in order[dim] if any(z['value']==v for z in q)];xx=np.arange(len(values))
  for j,a in enumerate(ARMS):
   zz=[next(z for z in q if z['arm']==a and z['value']==v) for v in values];axs[0,col].bar(xx+(j-1)*.25,[100*z['accuracy'] for z in zz],.25,color=COLORS[a],label=a);axs[1,col].bar(xx+(j-1)*.25,[z['mean_margin_over30'] for z in zz],.25,color=COLORS[a])
  ns=[next(z['count'] for z in q if z['value']==v) for v in values];labs=[v+'\nn='+str(n) for v,n in zip(values,ns)]
  for row in range(2):axs[row,col].set(xticks=xx,xticklabels=labs,title='Low urban: '+('quality proxy (dB)' if dim=='quality' else dim));axs[row,col].axhline(0,color='black',lw=.6)
 axs[0,0].set_ylabel('Accuracy (%)');axs[1,0].set_ylabel('Default-head margin / 30');axs[0,0].legend();save(fig,'low_urban_strata')
 fig,axs=plt.subplots(3,1,figsize=(12,10));xx=np.arange(7)
 for j,a in enumerate(ARMS):
  vals=[100*s['retention_on_fixed_RAW_clean_correct'][v][a]['accuracy'] for v in VIEWS];axs[0].bar(xx+(j-1)*.25,vals,.25,color=COLORS[a],label=a)
 axs[0].set(xticks=xx,xticklabels=LABELS,ylabel='Accuracy on fixed clean-correct set (%)',ylim=(0,105),title='Fixed RAW clean-correct n='+str(s['fixed_RAW_clean_correct_count']));axs[0].legend()
 for ax,ref in zip(axs[1:],('RAW','REPLAY')):
  pp=[z for z in s['paired'] if z['reference']==ref];ax.bar(xx-.18,[z['repairs'] for z in pp],.36,label=ref+' error repaired',color='#428367');ax.bar(xx+.18,[-z['breaks'] for z in pp],.36,label=ref+' correct broken',color='#b35252');ax.axhline(0,color='black',lw=.8);ax.set(xticks=xx,xticklabels=LABELS,ylabel='Held L packets',title='REFRESH versus '+ref+' | all n=1260');ax.legend()
 save(fig,'fixed_clean_retention_and_errors')
 return paths
