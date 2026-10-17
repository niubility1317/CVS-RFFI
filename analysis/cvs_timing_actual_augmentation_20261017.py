"""Read complete source telemetry;no IQ/checkpoint/truth/model or updates."""
import argparse,csv,json,math
from pathlib import Path
RUN='20261017-phase1-FcrossRX-timing-factorial-e30-s2026092701-m2-r01'
REFERENCE='20261017-phase1-FcrossRX-cycleorder-factorial-e30-s2026092701-m4-r01'
ORDER=('B00','B10','B01','B11')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
def extract(q):
 if q.get('head_fit_updates')!=0 or q.get('U_truth_read') is not False or q.get('target_access') is not False:raise ValueError('Source-only jointly trained default head required')
 records=q['records'];curves=[];moves=[]
 if [r['epoch'] for r in records]!=list(range(1,31)):raise ValueError('All30 source epoch records required')
 for r in records:
  e=r['epoch'];a=r['ordinary'];p=a['probabilities']
  if r['steps']!=e*222 or a['calls']!=e*222 or a['packets']!=e*222*128 or r['L_forward_packets']!=e*222*512 or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 or v>1 for v in p.values()):raise ValueError('Actual complete native augmentation schedule required')
  if not all(math.isfinite(a[k]) and a[k]>=0 for k in ('waveform_MSE','max_abs')):raise ValueError('Finite actual augmentation observations required')
  curves.append(dict(epoch=e,calls=a['calls'],packets=a['packets'],cumulative_waveform_MSE=a['waveform_MSE'],cumulative_max_abs=a['max_abs'],**p))
  for name,value in r['parameter_movement'].items():
   if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('Finite parameter movement required')
   moves.append(dict(epoch=e,parameter=name,L2_from_initial=value))
 return curves,moves

def compare(curves):
 ref=curves['B00'];answer={}
 for arm,rows in curves.items():
  if len(rows)!=30 or [r['epoch'] for r in rows]!=[r['epoch'] for r in ref]:raise ValueError('Same30 source epochs required')
  answer[arm]=dict(probability_epoch_mismatches=sum(any(a[k]!=b.get(k) for k in a if k.startswith('p_')) for a,b in zip(rows,ref)),waveform_observation_epoch_mismatches=sum(any(a[k]!=b[k] for k in ('cumulative_waveform_MSE','cumulative_max_abs','calls','packets')) for a,b in zip(rows,ref)))
 return answer

def csv_write(p,rows):
 with Path(p).open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def plots(out,curves,movements):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors=dict(B00='#64748b',B10='#16a34a',B01='#2563eb',B11='#dc2626');files=[]
 def save(fig,name):
  fig.tight_layout()
  for suffix in ('png','pdf'):
   p=out/(name+'.'+suffix);fig.savefig(p,dpi=200);files.append(str(p))
  plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13.5,4))
 keys=[k for k in curves['B00'][0] if k.startswith('p_')]
 for k in keys:axes[0].plot(range(1,31),[r[k] for r in curves['B00']],label=k[2:])
 for arm,rows in curves.items():
  axes[1].plot(range(1,31),[r['cumulative_waveform_MSE'] for r in rows],label=arm,color=colors[arm]);axes[2].plot(range(1,31),[r['cumulative_max_abs'] for r in rows],label=arm,color=colors[arm])
 for ax,title in zip(axes,('Actual old ordinary probabilities (B00 reference)','Cumulative ordinary-to-raw MSE','Cumulative max absolute ordinary IQ')):
  ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=7,ncol=2)
 fig.suptitle('Recorded shared ordinary augmentation; new timing is a separate secondary view');save(fig,'ordinary_augmentation_curves')
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for arm,rows in movements.items():
  for ax,ending in zip(axes,('cls_head.weight','sinc.low_coordinate','sinc.band_coordinate')):
   rr=[r for r in rows if r['parameter'].endswith(ending)]
   if len(rr)!=30:raise ValueError('Actual default head and Sinc30 updates required')
   ax.plot([r['epoch'] for r in rr],[r['L2_from_initial'] for r in rr],label=arm,color=colors[arm])
 for ax,title in zip(axes,('Default cosine head weight','Sinc low coordinate','Sinc band coordinate')):
  ax.set(title=title,xlabel='Epoch',ylabel='L2 distance from same-row initialization');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
 fig.suptitle('Joint update evidence; parameter movement is not identification performance');save(fig,'joint_training_movement');return files

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args();curves={};movements={};origins={}
 for arm in ORDER:
  run=RUN if arm in ('B10','B11') else REFERENCE;root=a.project/'runs'/run/(arm+'-s2026092701')/'source';done=read(root/'completion.json')
  if done['epoch']!=30 or done['optimizer_steps']!=6660 or done['target_access'] or done['head_fit_updates']:raise ValueError('Completed original fullsource E30 required')
  curves[arm],movements[arm]=extract(read(root/'mechanism_progress.json'));origins[arm]=dict(run=run,reused_control=arm in ('B00','B01'))
 comparison=compare(curves);out=a.output;out.mkdir(exist_ok=False)
 csv_write(out/'ordinary_probability_curves.csv',[dict(arm=arm,origin_run_id=origins[arm]['run'],reused_control=origins[arm]['reused_control'],**r) for arm,rows in curves.items() for r in rows]);csv_write(out/'parameter_movement.csv',[dict(arm=arm,**r) for arm,rows in movements.items() for r in rows]);files=plots(out,curves,movements)
 write(out/'summary.json',dict(code_commit=a.commit,origins=origins,ordinary_stream_comparison=comparison,ordinary_probability_ranges={k:[min(r[k] for r in curves['B00']),max(r[k] for r in curves['B00'])] for k in curves['B00'][0] if k.startswith('p_')},records=120,parameter_records=sum(map(len,movements.values())),NN=0,IQ_reads=0,CP_reads=0,truth_reads=0,new_updates=0,perpacket_gate_activations_saved=False,perbranch_gradients_saved=False,parameter_movement_proves_accuracy=False));write(out/'completion.json',dict(status='SOURCE_TELEMETRY_ANALYZED',records=120,figures=len(files),files=files,NN=0,IQ_reads=0,CP_reads=0,truth_reads=0,new_updates=0,code_commit=a.commit));print(json.dumps(dict(status='SOURCE_TELEMETRY_ANALYZED',out=str(out),files=len(files),NN=0)))
if __name__=='__main__':main()
