"""Frozen per-ID source decisions, complete same-row stratification and repairs."""
import csv
from pathlib import Path
import numpy as np
from . import design as d

PAIRS=(('AS_EMA','NOAS_EMA'),('AS_STUDENT','NOAS_STUDENT'),('AS_EMA','AS_STUDENT'),('NOAS_EMA','NOAS_STUDENT'))

def metric(y,p):
 y=np.asarray(y,np.int64);p=np.asarray(p,np.int64)
 if y.ndim!=1 or p.shape!=y.shape or not len(y) or np.any((y<0)|(y>=6)|(p<0)|(p>=6)):raise ValueError('Six-class paired decisions required')
 cm=np.bincount(y*6+p,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1)
 f1=np.divide(2*np.diag(cm),den,out=np.zeros(6,dtype=np.float64),where=den>0)
 hits=int((y==p).sum())
 return dict(count=len(y),hits=hits,errors=len(y)-hits,accuracy=hits/len(y),macro_f1=float(f1.mean()),confusion=cm.tolist())

def rows(label,scope,scene,draw,y,p,rx,day):
 rx=np.asarray(rx);day=np.asarray(day);records=[]
 dimensions=[('overall',[('ALL',np.ones(len(y),bool))]),('RX',[(str(v),rx==v) for v in sorted(set(rx))]),('day',[(str(v),day==v) for v in sorted(set(day))]),('TX',[(str(v),y==v) for v in range(6)]),('RX_day',[(str(r)+':'+str(v),(rx==r)&(day==v)) for r in sorted(set(rx)) for v in sorted(set(day))])]
 for dim,masks in dimensions:
  for stratum,mask in masks:
   if mask.any():records.append(dict(readout=label,scope=scope,scene=scene,draw=draw,dimension=dim,stratum=stratum,**metric(y[mask],p[mask])))
 result=metric(y,p);result['worst_RX_accuracy']=min(metric(y[rx==v],p[rx==v])['accuracy'] for v in sorted(set(rx)))
 return records,result

def paired(y,before,after):
 a=after==y;b=before==y;repairs=int((a&~b).sum());new=int((~a&b).sum())
 return dict(count=len(y),repairs=repairs,new_errors=new,net_repairs=repairs-new,accuracy_delta_pp=100*(repairs-new)/len(y))

def summarize(full,panel,out):
 records=[];full_summary={};panel_summary={};full_pairs={};panel_pairs={}
 for label in d.LABELS:
  r,m=rows(label,'fullV','clean',-1,full['TX'],full[label].argmax(-1),full['RX'],full['day']);records+=r;full_summary[label]=m;panel_summary[label]={}
  observations=[('clean',-1,panel[label]['clean'],panel['TX'],panel['RX'],panel['day'])]
  for si,scene in enumerate(d.SCENES):
   for draw in (-1,0,1):
    idx=(0,1) if draw==-1 else (draw,)
    observations.append((scene,draw,np.concatenate([panel[label]['satellite'][j,si] for j in idx]),np.tile(panel['TX'],len(idx)),np.tile(panel['RX'],len(idx)),np.tile(panel['day'],len(idx))))
  for scene,draw,logits,y,rx,day in observations:
   r,m=rows(label,'V1800_panel',scene,draw,y,logits.argmax(-1),rx,day);records+=r
   if draw==-1:panel_summary[label][scene]=m
 for a,b in PAIRS:
  name=a+'_minus_'+b;full_pairs[name]=paired(full['TX'],full[b].argmax(-1),full[a].argmax(-1));panel_pairs[name]={}
  for scene in ('clean',*d.SCENES):
   si=d.SCENES.index(scene) if scene!='clean' else None
   draws=(-1,) if scene=='clean' else (-1,0,1)
   for draw in draws:
    if scene=='clean':y=panel['TX'];before=panel[b]['clean'];after=panel[a]['clean']
    else:
     idx=(0,1) if draw==-1 else (draw,);y=np.tile(panel['TX'],len(idx))
     before=np.concatenate([panel[b]['satellite'][j,si] for j in idx]);after=np.concatenate([panel[a]['satellite'][j,si] for j in idx])
    panel_pairs[name].setdefault(scene,{})[str(draw)]=paired(y,before.argmax(-1),after.argmax(-1))
 interactions=dict(fullV_clean_pp=full_pairs['AS_EMA_minus_AS_STUDENT']['accuracy_delta_pp']-full_pairs['NOAS_EMA_minus_NOAS_STUDENT']['accuracy_delta_pp'],panel={s:panel_pairs['AS_EMA_minus_AS_STUDENT'][s]['-1']['accuracy_delta_pp']-panel_pairs['NOAS_EMA_minus_NOAS_STUDENT'][s]['-1']['accuracy_delta_pp'] for s in ('clean',*d.SCENES)})
 fields=('readout','scope','scene','draw','dimension','stratum','count','hits','errors','accuracy','macro_f1')
 with (Path(out)/'metrics.csv').open('w',newline='',encoding='utf8') as stream:
  writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in records)
 d.write(Path(out)/'fullV_confusions.json',dict(records=[r for r in records if r['scope']=='fullV']))
 d.write(Path(out)/'paired_errors.json',dict(fullV=full_pairs,panel=panel_pairs,interaction=interactions))
 return dict(fullV=full_summary,panel=panel_summary,paired_fullV=full_pairs,paired_panel=panel_pairs,interaction=interactions,metric_rows=len(records))
