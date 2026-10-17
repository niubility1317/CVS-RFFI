"""Plot completed P43 source metadata;no IQ,weights,heads or target inputs."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ARMS=('B01','B11_WHITE','B11_BG');RX=(1,3,4,6,8);SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban');STATES=('LOS','PARTIAL','BLOCKED');LOCKS=('locked','degraded','unlocked')
GRID=[(t,r,d,s,k)for t in range(6)for r in RX for d in(1,2,3)for s in SCENES for k in STATES];INDEX={k:i for i,k in enumerate(GRID)}
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def run(coverage,source,out):
 tic=time.perf_counter();proof=read(coverage/'complete.json');src=read(source/'source_summary.json')
 if proof['status']!='VERIFIED_SOURCE_TRAINING_STATE_COVERAGE'or proof['analysis_NN']or proof['target_access']or src['target_access']or src['NN']:raise ValueError('Completed legal source-derived metadata only')
 counts={a:np.zeros(810,dtype=np.int64)for a in ARMS};epochs={a:np.zeros((30,810),dtype=np.int64)for a in ARMS};seen=set()
 with(coverage/'TX_RX_day_scene_state.csv').open(encoding='utf8',newline='')as f:
  for q in csv.DictReader(f):
   a=q['arm'];key=(int(q['TX']),int(q['RX']),int(q['day']),q['scene'],q['state']);handle=(a,key);n=int(q['count'])
   if a not in ARMS or key not in INDEX or handle in seen or n<=0:raise ValueError('Invalid/duplicate actual exposure')
   counts[a][INDEX[key]]=n;seen.add(handle)
 if len(seen)!=3*810:raise ValueError('All810 cells in all3 As rows required')
 seen=set()
 with(coverage/'epoch_TX_RX_state.csv').open(encoding='utf8',newline='')as f:
  for q in csv.DictReader(f):
   a=q['arm'];e=int(q['epoch']);key=(int(q['TX']),int(q['RX']),int(q['day']),q['scene'],q['state']);handle=(a,e,key);n=int(q['count'])
   if a not in ARMS or not 1<=e<=30 or key not in INDEX or handle in seen or n<=0:raise ValueError('Invalid/duplicate actual epoch exposure')
   epochs[a][e-1,INDEX[key]]=n;seen.add(handle)
 for a in ARMS:
  if int(counts[a].sum())!=426240 or int(counts[a].min())!=proof['rows'][a]['minimum_cell_count']or not np.array_equal(epochs[a].sum(0),counts[a])or not(epochs[a].sum(1)==14208).all():raise ValueError('CSV counts disagree with actual complete logs')
 if not all(np.array_equal(counts[a],counts['B01'])and np.array_equal(epochs[a],epochs['B01'])for a in ARMS):raise ValueError('Do not merge different row exposures')
 if out.exists():raise FileExistsError('Preserve previous figures')
 out.mkdir();files=[];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'savefig.facecolor':'white'})
 def save(fig,name):
  for ext in('png','pdf'):p=out/(name+'.'+ext);fig.savefig(p,dpi=180,bbox_inches='tight');files.append(str(p))
  plt.close(fig)
 vals=counts['B01'];mx=max(sum(int(vals[INDEX[(t,r,d,s,k)]])for d in(1,2,3))for t in range(6)for r in RX for s in SCENES for k in STATES);fig,axes=plt.subplots(1,3,figsize=(12,11),sharey=True,layout='constrained');labels=['TX'+str(t)+'/RX'+str(r)for t in range(6)for r in RX]
 for s,ax in zip(SCENES,axes):
  data=np.asarray([[sum(int(vals[INDEX[(t,r,d,s,k)]])for d in(1,2,3))for k in STATES]for t in range(6)for r in RX]);im=ax.imshow(np.log1p(data),vmin=0,vmax=np.log1p(mx),cmap='Blues',aspect='auto');ax.set(xticks=range(3),xticklabels=STATES,yticks=range(30),yticklabels=labels,title=s.replace('practical_',''));ax.tick_params(axis='y',labelsize=7)
  for i in range(30):
   for j in range(3):ax.text(j,i,str(data[i,j]),ha='center',va='center',fontsize=7,color='white'if np.log1p(data[i,j])>.58*np.log1p(mx)else'#202735')
 cb=fig.colorbar(im,ax=axes,shrink=.5,pad=.025);ticks=np.linspace(0,np.log1p(mx),5);cb.set_ticks(ticks,labels=[str(round(np.expm1(v)))for v in ticks]);cb.set_label('Actual observations; log1p color');fig.suptitle('Identical exposure in B01/B11_WHITE/B11_BG | all6 TX x all5 source RX | days1-3 summed');save(fig,'all_TX_RX_state_exposure')
 cumulative=np.cumsum(epochs['B01'],0);fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained');axes[0].plot(range(1,31),100*np.count_nonzero(cumulative,axis=1)/810,'o-',ms=3);axes[0].set(ylabel='Covered cells (%)',ylim=(0,104),title='810 TX x RX x day x scene x state cells');axes[1].plot(range(1,31),cumulative.min(1),'o-',ms=3);axes[1].set(ylabel='Observations in least exposed cell',title='Cumulative minimum actual As count')
 for ax in axes:ax.set(xlabel='Cumulative epoch',xlim=(1,30));ax.grid(alpha=.18)
 fig.suptitle('All3 As-on rows verified equal; As-off rows have0 channel generation');save(fig,'all_state_support_progress')
 rows=src['rows'];cells=[s+'/'+l for s in STATES for l in LOCKS if any(q['scene']in SCENES[1:]and q['draw']==-1 and q['arm']=='B00'and q['dimension']=='state_lock'and q['stratum']==s+'/'+l for q in rows)];fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained');colors=('#64748b','#2563eb','#d97706')
 for i,scene in enumerate(SCENES[1:]):
  ns=[next((q['count']for q in rows if(q['arm'],q['scene'],q['draw'],q['dimension'],q['stratum'])==('B00',scene,-1,'state_lock',s)),0)for s in cells]
  if sum(ns)!=3600:raise ValueError('Every source state sample must be shown')
  for j,names in enumerate((('B00','B10_WHITE','B10_BG'),ARMS)):
   ax=axes[i,j]
   for a,color in zip(names,colors):
    v=[next((100*q['accuracy']for q in rows if(q['arm'],q['scene'],q['draw'],q['dimension'],q['stratum'])==(a,scene,-1,'state_lock',s)),np.nan)for s in cells];ax.plot(range(len(cells)),v,'o-',label=a,color=color)
   ax.set(xticks=range(len(cells)),xticklabels=[s.replace('/','\n')+'\nn='+str(n)for s,n in zip(cells,ns)],ylim=(0,100),title=scene.replace('practical_','')+(' / As off'if j==0 else' / As on'),ylabel='Source accuracy (%)');ax.tick_params(axis='x',labelsize=8);ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.18)
 fig.suptitle('Every nonempty source state/lock cell | V1800 x2 draws | tiny n is descriptive only; no target state attribution');save(fig,'source_all_nonempty_state_lock')
 result=dict(status='PLOTTED_COMPLETE_SOURCE_COVERAGE',figures=files,all3_exposure_arrays_equal=True,all810_cells_per_row=True,minimum_cell_count=int(counts['B01'].min()),source_state_cells=cells,source_samples_per_urban_scene=3600,analysis_NN=0,IQ_reads=0,checkpoint_reads=0,target_access=False,wall_seconds=time.perf_counter()-tic);(out/'complete.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--coverage',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.coverage,a.source,a.output)))
