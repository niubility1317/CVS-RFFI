"""Independent all-record loss/margin/cosine recount;no model or IQ read."""
import json,csv,math,time
from pathlib import Path
import numpy as np
import torch
from . import design as d

def array(t):return np.asarray(t.detach().cpu().tolist())
def angle(a,b):
 if a.ndim!=2 or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Finite paired actual gradient vectors required')
 a=a.astype(np.float64);b=b.astype(np.float64);na=np.linalg.norm(a,axis=1);nb=np.linalg.norm(b,axis=1);valid=(na>1e-12)&(nb>1e-12);v=np.zeros(len(a));v[valid]=np.clip(np.einsum('ij,ij->i',a[valid],b[valid])/(na[valid]*nb[valid]),-1,1);return v,valid,na,nb
def decision(cells):
 if len(cells)!=2 or sorted(x['draw'] for x in cells)!=[0,1]:raise ValueError('Both draws required')
 checks=[x['N']==900 and x['valid_full']>=855 and x['valid_no_noise']>=855 and x['negative_full']>=180 and x['negative_full']-x['negative_no_noise']>=90 for x in sorted(cells,key=lambda r:r['draw'])]
 return dict(status='SOURCE_GRADIENT_DECIDED',margin_noise_conflict_evidence=all(checks),draw_checks=checks,primary_cells=cells,source_only=True,V_backprop=False,target_access=False,training_updates=0,head_fit_updates=0,automatic_training=False,Ag_proved=False,goal_complete=False)
def write_csv(path,rows):
 with Path(path).open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metric(y,z):
 p=z.argmax(1);cm=np.zeros((6,6),int);np.add.at(cm,(y,p),1);den=cm.sum(0)+cm.sum(1);f1=np.divide(2*cm.diagonal(),den,out=np.zeros(6),where=den>0);other=z.copy();other[np.arange(len(y)),y]=-np.inf;margin=(z[np.arange(len(y)),y]-other.max(1))/30
 return dict(N=len(y),accuracy=float((p==y).mean()),macro_f1=float(f1.mean()),margin_over30_mean=float(margin.mean()),confusion=cm.tolist())
def collect(root):
 tic=time.perf_counter();root=Path(root);done=d.read(root/'observation_complete.json');roles=d.read(root/'data_roles.json')
 if done['NN']!=11702 or done['training_updates'] or done['head_fit_updates'] or done['target_access'] or done['V_backprop'] or roles['role']!='L_s' or roles['source_physical_count']!=900:raise ValueError('Only complete lawful L-only observations')
 meta=torch.load(root/'source_labels_ids.pt',map_location='cpu',weights_only=False);y=array(meta['y']).astype(int);rx=array(meta['rx']);day=array(meta['day']);clean=torch.load(root/'clean.pt',map_location='cpu',weights_only=False);base_logits=array(clean['logits']);clean_correct=base_logits.argmax(1)==y;rows=[];metrics=[];gate_rows=[];primary={}
 for draw in (0,1):
  for scene in d.SCENES:
   p=root/f'draw{draw}'/scene;physics=d.read(p/'physics.json');full=[r['full'] for r in physics];state=np.asarray([r['state'] for r in full]);lock=np.asarray([r['receiver_lock_status'] for r in full]);quality=np.asarray([r['signal_to_added_noise_db'] for r in full],float)
   masks=[('overall','ALL',np.ones(900,bool))]+[('RX',str(r),rx==r) for r in d.RX]+[('day',str(v),day==v) for v in d.DAYS]+[('TX',str(v),y==v) for v in range(6)]+[('state',v,state==v) for v in ['LOS','PARTIAL','BLOCKED']]+[('lock',v,lock==v) for v in ['locked','degraded','unlocked']]+[('added_noise_ratio',name,(quality>=lo)&(quality<hi)) for name,lo,hi in [('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)]]
   for condition in d.CONDITIONS:
    actual=torch.load(p/(condition.replace('/','_')+'.pt'),map_location='cpu',weights_only=False);z=array(actual['logits'])
    if z.shape!=(900,6) or not np.isfinite(z).all():raise ValueError('Finite original default logits required')
    shifted=z-z.max(1)[:,None];logp=shifted-np.log(np.exp(shifted).sum(1))[:,None];ce=-logp[np.arange(900),y];others=z.copy();others[np.arange(900),y]=-np.inf;mar=(z[np.arange(900),y]-others.max(1))/30
    for name,v in [('CE0',ce),('CE01',.99*ce-.01*logp.mean(1)),('margin',-mar)]:
     if not np.allclose(array(actual['losses'][name]),v,atol=3e-5,rtol=3e-5):raise ValueError('Independent objective definition differs '+name)
    for dim,label,mask in masks:
     if mask.any():
      cohort=mask&clean_correct;metrics.append(dict(draw=draw,scene=scene,condition=condition,dimension=dim,stratum=label,fixed_clean_correct_N=int(cohort.sum()),fixed_clean_correct_retention=float((z.argmax(1)[cohort]==y[cohort]).mean()) if cohort.any() else None,**metric(y[mask],z[mask])))
    for objective in ('margin','CE0','CE01'):
     for kind,key in [('parameter_gradients','parameter'),('feature_gradients','feature')]:
      for name,t in actual['objectives'][objective][kind].items():
       cos,valid,n0,n1=angle(array(clean['objectives'][objective][kind][name]),array(t))
       for dim,label,mask in masks:
        n=int(mask.sum())
        if not n:continue
        ok=valid&mask;neg=int(((cos<0)&ok).sum());r=dict(draw=draw,scene=scene,condition=condition,objective=objective,kind=key,name=name,dimension=dim,stratum=label,N=n,valid=int(ok.sum()),undefined=n-int(ok.sum()),negative=neg,negative_fraction_all=neg/n,cosine_mean_valid=float(cos[ok].mean()) if ok.any() else None,clean_gradient_norm_mean=float(n0[mask].mean()),view_gradient_norm_mean=float(n1[mask].mean()));rows.append(r)
        if scene=='practical_low_urban' and objective=='margin' and kind=='parameter_gradients' and name=='head_W' and dim=='overall':primary[(draw,condition)]=r
    for gate,g in actual['gates'].items():
     diff=array(g['mean'])-array(clean['gates'][gate]['mean']);gate_rows.append(dict(draw=draw,scene=scene,condition=condition,gate=gate,mean_change=float(diff.mean()),absolute_mean_change=float(abs(diff).mean()),observations=900))
 cells=[]
 for draw in (0,1):
  f=primary[(draw,'full')];n=primary[(draw,'no_noise/fixed')];cells.append(dict(draw=draw,objective='margin',parameter='head_W',N=900,valid_full=f['valid'],valid_no_noise=n['valid'],negative_full=f['negative'],negative_no_noise=n['negative']))
 decision_result=decision(cells)
 from .criteria import decide
 if decide(cells)['margin_noise_conflict_evidence']!=decision_result['margin_noise_conflict_evidence'] or decide(cells)['draw_checks']!=decision_result['draw_checks']:raise ValueError('Independent source rule differs')
 out=root/'analysis';out.mkdir(exist_ok=False);write_csv(out/'paired_gradient_strata.csv',rows);write_csv(out/'gate_changes.csv',gate_rows);write_csv(out/'source_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in metrics]);d.write(out/'confusion_metrics.json',metrics)
 result=dict(status='ANALYZED_SOURCE_ONLY',source_decision=decision_result,rows=len(rows),metric_confusions=len(metrics),clean_metric=metric(y,base_logits),fixed_clean_correct_count=int(clean_correct.sum()),NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0,V_backprop=False,U_truth_read=False,target_access=False,head_fit_updates=0,scope='Frozen local-point gradients,not actual historical training trajectory;LS gradients cannot alone show identity conflict',model_seed_count=1,channel_draws=2,new_Ag_proved=False,automatic_training=False,wall_seconds=time.perf_counter()-tic,goal_complete=False)
 from .plots import make
 result['figures']=make(out,rows,metrics);d.write(out/'summary.json',result);return result
