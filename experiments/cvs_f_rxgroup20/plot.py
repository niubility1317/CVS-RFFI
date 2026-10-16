"""Final source-only scientific figures;aggregates only,no model/IQ access."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ARMS=('BATCH','RXGROUP');COLORS={'BATCH':'#587894','RXGROUP':'#d77439'}
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
  vals=[100*s['scores'][a][v]['accuracy'] for v in VIEWS];bars=axs[0].bar(x+(j-.5)*.36,vals,.36,label=a,color=COLORS[a])
  axs[0].bar_label(bars,fmt='%.2f',fontsize=9,padding=2)
 axs[0].set(xticks=x,xticklabels=LABELS,ylabel='Accuracy (%)',ylim=(0,105),title='Both newly trained matched controls; all seven fixed readouts');axs[0].legend()
 gain=[s['all_view_gain_pp'][v] for v in VIEWS];bars=axs[1].bar(x,gain,color=['#428367' if z>=0 else '#b35252' for z in gain]);axs[1].bar_label(bars,fmt='%+.2f',padding=3,fontsize=9);axs[1].axhline(0,color='black',lw=.8);axs[1].set(xticks=x,xticklabels=LABELS,ylabel='RXGROUP - BATCH (pp)',title='Source qualification: '+str(s['qualified'])+'; no formal target or Ag+As evidence');save(fig,'source_scene_results')
 fig,axs=plt.subplots(3,2,figsize=(13,11));panels=(('total_L','Complete labeled objective'),('ordinary_CE','Ordinary-view CE'),('seenV_accuracy','Seen-RX V accuracy (%)'),('supcon','Original SupCon'),('grad_before_clip','Gradient norm before clipping'),('pseudo_selected','Original U accepted per epoch'))
 for ax,(k,title) in zip(axs.flat,panels):
  for a in ARMS:
   q=sorted((z for z in curves if z['arm']==a),key=lambda z:z['epoch']);ax.plot([z['epoch'] for z in q],[z[k] for z in q],marker='.',color=COLORS[a],label=a)
  ax.axvline(13.5,color='#999999',ls=':',lw=1);ax.set(title=title,xlabel='Epoch',xticks=(1,5,10,13,15,20));ax.grid(alpha=.2)
 axs[0,0].legend();save(fig,'complete_training_curves')
 fig,axs=plt.subplots(2,2,figsize=(13,9));keys=list(s['training']['BATCH']['actual_operator_packet_counts']);short=[k.replace('p_','') for k in keys];yy=np.arange(len(keys))
 for j,a in enumerate(ARMS):
  h=s['training'][a];rate=[100*h['actual_operator_packet_counts'][k]['actual_on']/h['actual_operator_packet_counts'][k]['eligible'] for k in keys];axs[0,0].barh(yy+(j-.5)*.36,rate,.36,label=a,color=COLORS[a]);rate=[100*h['pair_gate_counts'][k]['actual_disagree']/h['pair_gate_counts'][k]['positive_pairs'] for k in keys];axs[0,1].barh(yy+(j-.5)*.36,rate,.36,label=a,color=COLORS[a]);q=sorted((z for z in curves if z['arm']==a),key=lambda z:z['epoch']);axs[1,0].plot([z['epoch'] for z in q],[z['ordinary_MSE'] for z in q],label=a,color=COLORS[a]);axs[1,1].plot([z['epoch'] for z in q],[z['group_stage_MSE'] for z in q],label=a,color=COLORS[a])
 for ax in axs[0]:ax.set(yticks=yy,yticklabels=short);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
 axs[0,0].set(xlabel='Triggered L packets (%)',title='Realized marginal counts need not match');axs[0,0].legend();axs[0,1].set(xlabel='Same-TX cross-RX positive pairs (%)',title='Different gate states across a positive pair')
 axs[1,0].set(xlabel='Epoch',ylabel='MSE',title='Actual ordinary versus raw IQ');axs[1,1].set(xlabel='Epoch',ylabel='MSE',title='Stage-local group versus whole shadow');axs[1,1].text(.02,.95,'Stage-local shadow; not a full-pipeline counterfactual',transform=axs[1,1].transAxes,va='top',fontsize=8);save(fig,'ordinary_coupling_exposure')
 fig,axs=plt.subplots(2,3,figsize=(15,9))
 for col,dim in enumerate(('state','lock','quality')):
  q=[z for z in strata if z['scene']=='practical_low_urban' and z['dimension']==dim];order={'state':('LOS','PARTIAL','BLOCKED'),'lock':('locked','degraded','unlocked'),'quality':('<0','[0,5)','[5,10)','[10,20)','[20,30)','>=30')};values=[v for v in order[dim] if any(z['value']==v for z in q)];xx=np.arange(len(values))
  for j,a in enumerate(ARMS):
   zz=[next(z for z in q if z['arm']==a and z['value']==v) for v in values];axs[0,col].bar(xx+(j-.5)*.35,[100*z['accuracy'] for z in zz],.35,color=COLORS[a],label=a);axs[1,col].bar(xx+(j-.5)*.35,[z['mean_margin_over30'] for z in zz],.35,color=COLORS[a])
  ns=[next(z['count'] for z in q if z['value']==v) for v in values];labs=[v+'\nn='+str(n) for v,n in zip(values,ns)]
  for row in range(2):axs[row,col].set(xticks=xx,xticklabels=labs,title='Low urban: '+('quality proxy (dB)' if dim=='quality' else dim));axs[row,col].axhline(0,color='black',lw=.6)
 axs[0,0].set_ylabel('Accuracy (%)');axs[1,0].set_ylabel('Default-head margin / 30');axs[0,0].legend();save(fig,'low_urban_strata')
 fig,axs=plt.subplots(2,1,figsize=(12,8));xx=np.arange(7)
 for j,a in enumerate(ARMS):
  vals=[100*s['retention_on_fixed_Batch_clean_correct'][v][a]['accuracy'] for v in VIEWS];axs[0].bar(xx+(j-.5)*.36,vals,.36,color=COLORS[a],label=a)
 axs[0].set(xticks=xx,xticklabels=LABELS,ylabel='Accuracy on fixed clean-correct set (%)',ylim=(0,105),title='Fixed BATCH clean-correct n='+str(s['fixed_Batch_clean_correct_count']));axs[0].legend()
 pp=s['paired'];axs[1].bar(xx-.18,[z['repairs'] for z in pp],.36,label='BATCH error repaired',color='#428367');axs[1].bar(xx+.18,[-z['breaks'] for z in pp],.36,label='BATCH correct broken',color='#b35252');axs[1].axhline(0,color='black',lw=.8);axs[1].set(xticks=xx,xticklabels=LABELS,ylabel='Number of held L packets',title='All n=1260; positive and negative error changes');axs[1].legend();save(fig,'fixed_clean_retention_and_errors')
 return paths
