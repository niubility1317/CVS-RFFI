"""Offline source physics plots;no IQ/checkpoint/NN reads."""
import argparse,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .analyze import SCENES
LABELS=('High suburban','Mid suburban','Low suburban','High urban','Mid urban','Low urban')
def run(root,out):
 root=Path(root);out=Path(out);out.mkdir(exist_ok=False)
 a=json.loads((root/'independent_analysis.json').read_text());p=json.loads((root/'quality_plot_points.json').read_text());g=a['groups'];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
 manifest=[]
 def save(fig,name):
  fig.savefig(out/(name+'.png'),bbox_inches='tight',dpi=170);fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig);manifest.append(name)
 fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
 panels=[('median_abs_bias_db','Median |waveform - proxy| (dB)',1.,False),('whole_lock_flip_fraction','Receiver lock changes (%)',5.,True),('median_whole_vs_interior_abs_db','Median |whole - interior| (dB)',.5,False),('boundary_robust_flip_fraction','Whole flips retained by interior (%)',50.,True)]
 for ax,(key,title,threshold,percent) in zip(axs.flat,panels):
  for d,color in ((0,'#2878b5'),(1,'#e07b39')):
   vals=[g[str(d)+'/'+s][key]*(100 if percent else 1) for s in SCENES];ax.bar(np.arange(6)+(d-.5)*.34,vals,width=.32,color=color,label='Physics draw '+str(d))
  ax.axhline(threshold,color='#a61c00',ls='--',lw=1,label='Registered urban criterion');ax.set_xticks(range(6),LABELS,rotation=25,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axs[0,0].legend(fontsize=8);fig.suptitle('Source fit L5040 per scene/draw; two physics streams, no model training',fontsize=13);save(fig,'quality_criteria')
 fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
 for d in (0,1):
  for j,s in enumerate(SCENES[-2:]):
   ax=axs[d,j];v=[z['bias_db'] for z in p if z['draw']==d and z['scene']==s];ax.hist(v,bins=70,density=True,color='#2878b5',alpha=.85);ax.axvline(0,color='black',lw=1);ax.axvline(statistics.median(v),color='#c34a24',ls='--',label='Median %.3f dB'%statistics.median(v));ax.set_title('%s, draw%d, n=5040'%(LABELS[-2+j],d));ax.set_xlabel('Whole waveform quality - white-input proxy (dB)');ax.set_ylabel('Density');ax.legend(fontsize=8)
 fig.suptitle('Actual frozen-H finite-waveform power difference; denominator is added-noise expectation');save(fig,'urban_bias_distribution')
 statuses=('locked','degraded','unlocked');fig,axs=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
 for ax,s,label in zip(axs.flat,SCENES,LABELS):
  rows=[z for z in p if z['scene']==s];matrix=np.array([[sum(z['lock_proxy']==u and z['lock_whole']==v for z in rows) for v in statuses] for u in statuses]);ax.imshow(matrix/len(rows)*100,cmap='Blues',vmin=0,vmax=100)
  for i in range(3):
   for j in range(3):ax.text(j,i,'%d%s%.2f%%'%(matrix[i,j],chr(10),matrix[i,j]/len(rows)*100),ha='center',va='center',color='white' if matrix[i,j]/len(rows)>.5 else 'black',fontsize=9)
  ax.set_xticks(range(3),statuses,rotation=25);ax.set_yticks(range(3),statuses);ax.set_xlabel('Whole-waveform lock');ax.set_ylabel('Original proxy lock');ax.set_title(label)
 fig.suptitle('Lock confusion: 10080 records/scene = L5040 x two physics streams');save(fig,'lock_confusion')
 fig,axs=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
 for s,color,label in ((SCENES[-2],'#2878b5','Mid urban'),(SCENES[-1],'#e07b39','Low urban')):
  rates=[];cfo=[];boundary=[]
  for rx in (3,4,6,8):
   rows=[z for z in p if z['scene']==s and z['RX']==rx];rates.append(100*sum(z['lock_proxy']!=z['lock_whole'] for z in rows)/len(rows));cfo.append(statistics.mean(abs(z['residual_CFO_whole_hz']-z['residual_CFO_native_hz']) for z in rows));boundary.append(statistics.median(abs(z['whole_vs_interior_db']) for z in rows))
  for ax,v in zip(axs,(rates,cfo,boundary)):ax.plot(range(4),v,'o-',color=color,label=label);ax.set_xticks(range(4),['RX3','RX4','RX6','RX8']);ax.grid(axis='y',alpha=.2)
 for ax,title in zip(axs,('Lock flips (%)','Mean absolute residual CFO change (Hz)','Median whole/interior difference (dB)')):ax.set_title(title)
 axs[0].legend();fig.suptitle('Source receiver strata: 2520 records/RX/scene, same fixed hardware sessions');save(fig,'urban_receiver_strata')
 (out/'manifest.json').write_text(json.dumps(dict(status='CREATED',figures=manifest,source_only=True,physical_records=len(p),real_source_L=5040,NN=0,target_access=False),indent=2)+chr(10))
 return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.input,a.output)))
