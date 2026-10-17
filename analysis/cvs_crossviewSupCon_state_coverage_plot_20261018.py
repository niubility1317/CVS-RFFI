"""Plot completed metadata-only P41 exposure; never reads IQ, weights or truth."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ARMS=('ORD128','DUP192','SAT192');RX=(1,3,4,6,8);SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban');STATES=('LOS','PARTIAL','BLOCKED')
GRID=[(t,r,d,s,k) for t in range(6) for r in RX for d in (1,2,3) for s in SCENES for k in STATES]
INDEX={k:i for i,k in enumerate(GRID)}
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def load(root):
 proof=read(root/'complete.json')
 # Check the observed per-row evidence, never infer success from a filename.
 if proof['status']!='VERIFIED_SOURCE_TRAINING_STATE_COVERAGE' or proof['analysis_NN']!=0 or proof['target_access']:raise ValueError('Source-only completed coverage required')
 counts={a:np.zeros((len(GRID),2),dtype=np.int64) for a in ARMS};seen=set()
 with (root/'TX_RX_day_scene_state.csv').open(encoding='utf8',newline='') as f:
  for q in csv.DictReader(f):
   a=q['arm'];key=(int(q['TX']),int(q['RX']),int(q['day']),q['scene'],q['state']);handle=(a,key)
   if a not in counts or key not in INDEX or handle in seen:raise ValueError('Duplicate or out-of-contract exposure cell')
   n,j=int(q['count']),int(q['SupCon_active_count'])
   if n<=0 or not 0<=j<=n:raise ValueError('Invalid observation counts')
   counts[a][INDEX[key]]=(n,j);seen.add(handle)
 epochs={a:np.zeros((30,len(GRID)),dtype=np.int64) for a in ARMS};seen=set()
 with (root/'epoch_TX_RX_state.csv').open(encoding='utf8',newline='') as f:
  for q in csv.DictReader(f):
   a=q['arm'];e=int(q['epoch']);key=(int(q['TX']),int(q['RX']),int(q['day']),q['scene'],q['state']);handle=(a,e,key);n=int(q['count'])
   if a not in epochs or not 1<=e<=30 or key not in INDEX or handle in seen or n<=0:raise ValueError('Invalid complete epoch exposure')
   epochs[a][e-1,INDEX[key]]=n;seen.add(handle)
 for a in ARMS:
  z=proof['rows'][a]
  if z['U_TX_truth_read'] or z['identity_NN'] or z['target_access'] or z['known_L6300_only'] is not True:raise ValueError('Only legal L metadata coverage')
  if list(counts[a].sum(0))!=[z['actual_As_observations'],z['active_SupCon_As_observations']]:raise ValueError('CSV must agree with actual saved totals')
  if not np.array_equal(epochs[a].sum(0),counts[a][:,0]) or not np.array_equal(epochs[a][5:].sum(0),counts[a][:,1]) or not (epochs[a].sum(1)==14208).all():raise ValueError('Epoch and active totals must exactly match')
 return proof,counts,epochs

def run(root,out):
 tic=time.perf_counter();proof,counts,epochs=load(root)
 if out.exists():raise FileExistsError('Preserve independent prior plot output')
 out.mkdir();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
 identical=all(np.array_equal(counts[a],counts['ORD128']) and np.array_equal(epochs[a],epochs['ORD128']) for a in ARMS);display=('ORD128',) if identical else ARMS;files=[]
 def save(fig,name):
  for ext in ['png','pdf']:
   p=out/(name+'.'+ext);fig.savefig(p,dpi=180,bbox_inches='tight');files.append(str(p))
  plt.close(fig)
 for arm in display:
  values=counts[arm];maxv=max(sum(int(values[INDEX[(t,r,d,s,k)],0]) for d in (1,2,3)) for t in range(6) for r in RX for s in SCENES for k in STATES);fig,axes=plt.subplots(1,3,figsize=(11.8,11),sharey=True,layout='constrained');labels=['TX'+str(t)+'/RX'+str(r) for t in range(6) for r in RX]
  for si,(s,ax) in enumerate(zip(SCENES,axes)):
   data=np.asarray([[sum(int(values[INDEX[(t,r,d,s,k)],0]) for d in (1,2,3)) for k in STATES] for t in range(6) for r in RX]);im=ax.imshow(np.log1p(data),vmin=0,vmax=np.log1p(maxv),cmap='Blues',aspect='auto');ax.set(xticks=range(3),xticklabels=STATES,yticks=range(30),yticklabels=labels,title=s.replace('practical_','').replace('_',' '));ax.tick_params(axis='y',labelsize=7)
   for y in range(30):
    for x in range(3):ax.text(x,y,str(data[y,x]),ha='center',va='center',fontsize=7,color='white' if np.log1p(data[y,x])>.58*np.log1p(maxv) else '#202735')
   for y in [4.5,9.5,14.5,19.5,24.5]:ax.axhline(y,color='white',linewidth=1.5)
  cb=fig.colorbar(im,ax=axes,shrink=.46,pad=.025);ticks=np.linspace(0,np.log1p(maxv),5);cb.set_ticks(ticks,[str(round(np.expm1(v))) for v in ticks]);cb.set_label('Actual As observations; log1p color')
  fig.suptitle(('All three rows have identical exposure' if identical else arm)+' | 6 TX x 5 source RX | days 1-3 summed',fontsize=12);save(fig,'training_TX_RX_state_exposure'+('' if identical else '_'+arm))
 fig,axes=plt.subplots(1,2,figsize=(10.6,3.7),layout='constrained')
 colors={'ORD128':'#4169a1','DUP192':'#b16f28','SAT192':'#24805a'};series={}
 for a in display:
  regular=np.cumsum(epochs[a],0);active=epochs[a].copy();active[:5]=0;active=np.cumsum(active,0);series[a]={'CE_cells_per_epoch':np.count_nonzero(regular,axis=1).tolist(),'active_relationship_cells_per_epoch':np.count_nonzero(active,axis=1).tolist(),'minimum_CE_count_per_epoch':regular.min(1).tolist(),'minimum_active_count_per_epoch':active.min(1).tolist()};name='All three rows' if identical else a
  axes[0].plot(range(1,31),100*np.count_nonzero(regular,axis=1)/810,label=name+' / satellite CE',color=colors[a]);axes[0].plot(range(1,31),100*np.count_nonzero(active,axis=1)/810,'--',label=name+' / SupCon active epochs',color=colors[a]);axes[1].plot(range(1,31),regular.min(1),label=name+' / satellite CE',color=colors[a]);axes[1].plot(range(1,31),active.min(1),'--',label=name+' / SupCon active epochs',color=colors[a])
 axes[0].set(ylabel='Covered cells (%)',ylim=(0,104),title='810 TX x RX x day x scene x state cells');axes[1].set(ylabel='Observations in the least exposed cell',title='Cumulative minimum cell count')
 for ax in axes:ax.set(xlabel='Cumulative epoch',xlim=(1,30));ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=7)
 save(fig,'training_state_support_progress')
 summary=dict(status='PLOTTED_VERIFIED_SOURCE_COVERAGE',input=str(root),all_three_exposure_arrays_equal=identical,rows_shown=list(display),rows_verified=list(ARMS),series=series,figures=files,analysis_NN=0,IQ_reads=0,checkpoint_reads=0,U_TX_truth_read=False,target_access=False,training_changed=False,scope='Actual propagation state exposure from named-stream replay reconciled with native logs;receiver lock,noise,AGC and accuracy not recomputed',wall_seconds=time.perf_counter()-tic)
 (out/'complete.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8');return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.input,a.output)))
