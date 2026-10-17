"""Complete descriptive figures,one model seed/five historical controls;no NN."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
LABELS=('Clean','High suburban','Mid suburban','Low suburban','High urban','Mid urban','Low urban')
COLORS=('#0072B2','#D55E00','#009E73','#CC79A7','#333333')
def draw(root):
 root=Path(root);s=json.loads((root/'analysis_summary.json').read_text());curves=json.loads((root/'training_curves.json').read_text())
 if s['execution']!='VERIFIED' or not s['source_only'] or s['target_access'] or s['new_NN'] or s['source_saved_predictions_recounted']!=44100 or len(curves)!=100:raise ValueError('Complete fixed5 source audit first')
 out=root/'figures';out.mkdir(exist_ok=False);rows=s['rows'];keys=[r['key'] for r in rows];paths=[]
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True})
 def save(fig,name):
  fig.tight_layout()
  for ext in ('png','pdf'):
   p=out/(name+'.'+ext);fig.savefig(p,dpi=170,bbox_inches='tight');paths.append(p)
  plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(15,4.8))
 for ax,metric,title in zip(axes,('accuracy','macro_F1'),('Default-head accuracy (%)','Six-class macro F1 (%)')):
  arr=np.array([[100*r['scores'][v][metric] for v in VIEWS] for r in rows]);im=ax.imshow(arr,vmin=0,vmax=100,cmap='Blues',aspect='auto')
  ax.set_xticks(range(7),LABELS,rotation=40,ha='right');ax.set_yticks(range(5),keys);ax.set_title(title)
  for i in range(5):
   for j in range(7):ax.text(j,i,f'{arr[i,j]:.2f}',ha='center',va='center',color='white' if arr[i,j]>65 else 'black',fontsize=9)
 fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,label='Percent',fraction=.025,pad=.02);fig.suptitle('One exposed held-source RX1 / L1260; five historical controls are not five seeds',y=1.02)
 # Explicit layout keeps the shared colorbar separate from both heatmaps.
 fig.subplots_adjust(left=.09,right=.88,bottom=.27,wspace=.38,top=.86)
 for ext in ('png','pdf'):
  p=out/('historical_control_metrics.'+ext);fig.savefig(p,dpi=170,bbox_inches='tight');paths.append(p)
 plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12.5,5))
 for ax,view,title in zip(axes,('clean','practical_low_urban'),('Clean','Low urban')):
  mat=np.zeros((5,5))
  for p in s['pairs']:
   i,j=keys.index(p['first']),keys.index(p['second']);mat[i,j]=mat[j,i]=100*p['eval_views'][view]['hard_disagreement_fraction']
  im=ax.imshow(mat,vmin=0,vmax=7,cmap='Oranges');ax.set_xticks(range(5),keys,rotation=40,ha='right');ax.set_yticks(range(5),keys);ax.set_title(title)
  for i in range(5):
   for j in range(5):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if mat[i,j]>4.5 else 'black')
 fig.subplots_adjust(left=.12,right=.88,bottom=.24,wspace=.4,top=.85);fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='Hard prediction disagreement (%)',fraction=.03,pad=.02);fig.suptitle('Same held physical IDs and labels; cross-run exact IQ equality was not tested',y=.99)
 for ext in ('png','pdf'):
  p=out/('paired_hard_decisions.'+ext);fig.savefig(p,dpi=170,bbox_inches='tight');paths.append(p)
 plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(12.5,8))
 for i,key in enumerate(keys):
  rr=[q for q in curves if q['control']==key];x=[q['epoch'] for q in rr]
  if x!=list(range(1,21)):raise ValueError('All20 epochs of every control')
  for ax,metric in zip(axes.flat,('total_L','gradient_preclip','seenV_percent','pseudo_selected_epoch_SUM')):ax.plot(x,[q[metric] for q in rr],marker='.',color=COLORS[i],label=key,linewidth=1.3)
 for ax,title,ylabel in zip(axes.flat,('Full objective','Gradient before clipping','Seen-source V health','Original U selections'),('Mean loss / epoch','Mean L2 norm / update','Accuracy (%)','Selected packets (epoch SUM)')):
  ax.set_title(title);ax.set_xlabel('Epoch');ax.set_ylabel(ylabel);ax.set_xlim(1,20);ax.set_xticks([1,5,10,13,14,17,20]);ax.axvline(13.5,linestyle='--',color='gray',alpha=.7);ax.grid(alpha=.2)
 axes[0,1].set_yscale('log');axes[0,0].legend(fontsize=9);fig.suptitle('All 100 saved epoch rows; dashed line marks original L-only to old-gate U stage',y=1.01);save(fig,'complete_training_curves')
 (out/'manifest.json').write_text(json.dumps(dict(status='VERIFIED',source_only=True,target_access=False,new_NN=0,model_seeds=s['model_seeds'],historical_controls=5,figures=[p.name for p in paths]),indent=2)+'\n',encoding='utf8');return paths
