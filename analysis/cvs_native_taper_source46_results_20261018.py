"""Independent saved-L-logit recount; never imports candidate/model/IQ loaders."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
ARMS=('B00','B01');RX=(1,3,4,6,8)
def measure(z,y):
 p=z.argmax(1);cm=np.bincount(6*y+p,minlength=36).reshape(6,6);tp=np.diag(cm).astype(float);den=cm.sum(0)+cm.sum(1);f=np.divide(2*tp,den,out=np.zeros(6),where=den>0)
 zz=z.astype(np.float64);v=zz[np.arange(len(y)),y].copy();zz[np.arange(len(y)),y]=-np.inf
 return dict(count=len(y),correct=int(tp.sum()),accuracy=float(tp.sum()/len(y)),macro_f1=float(f.mean()),mean_margin_over30=float(((v-zz.max(1))/30).mean()),confusion=cm.tolist())
def calculate(raw,full,bypass,y,rx):
 if raw.shape!=(2,6300,6) or full.shape!=(2,2,6300,6) or bypass.shape!=full.shape or y.shape!=(6300,) or set(rx.tolist())!=set(RX) or set(y.tolist())!=set(range(6)) or not all(np.isfinite(a).all() for a in (raw,full,bypass)):raise ValueError('Saved same-source matrices differ')
 fixed=raw[0].argmax(1)==y
 if not fixed.any():raise ValueError('Fixed source cohort empty')
 overall=[];rxrows=[]
 for ai,arm in enumerate(ARMS):
  for dr in (0,1):
   old=full[ai,dr].argmax(1)==y;new=bypass[ai,dr].argmax(1)==y
   overall.append(dict(arm=arm,draw=dr,count=6300,FULL_correct=int(old.sum()),NO_TAPER_correct=int(new.sum()),FULL_retained=int(old[fixed].sum()),NO_TAPER_retained=int(new[fixed].sum()),retention_gain_pp=float(100*(new[fixed].sum()-old[fixed].sum())/fixed.sum()),accuracy_delta_pp=float(100*(new.sum()-old.sum())/6300),repairs=int(np.logical_and(new,~old).sum()),new_errors=int(np.logical_and(old,~new).sum())))
  for r in RX:
   mask=(rx==r)&fixed;old=full[ai].argmax(2)==y;new=bypass[ai].argmax(2)==y
   rxrows.append(dict(arm=arm,RX=r,count_per_draw=int(mask.sum()),retention_gain_pp=float(100*(new[:,mask].sum()-old[:,mask].sum())/(2*mask.sum()))))
 checks=dict(each_model_draw_retention=all(q['retention_gain_pp']>=.2-1e-10 for q in overall),each_model_draw_overall_nonnegative=all(q['accuracy_delta_pp']>=-1e-10 for q in overall),each_model_at_least4positive_RX=all(sum(q['retention_gain_pp']>0 for q in rxrows if q['arm']==a)>=4 for a in ARMS),each_model_worst_RX_preserved=all(q['retention_gain_pp']>=-.25-1e-10 for q in rxrows))
 return dict(overall=overall,per_RX=rxrows,checks=checks,source_qualified=all(checks.values()),cohort_count=int(fixed.sum()))
def selftest():
 y=np.arange(6300)%6;rx=np.repeat(np.array(RX),1260);z=np.eye(6,dtype=np.float32)[y]*30;raw=np.stack([z,z]);full=np.repeat(raw[:,None],2,axis=1);bypass=full.copy()
 assert not calculate(raw,full,bypass,y,rx)['source_qualified']
 for r in RX:
  ix=np.flatnonzero(rx==r)[:10];full[:,:,ix,:]=0;full[:,:,ix,(y[ix]+1)%6]=30
 assert calculate(raw,full,bypass,y,rx)['source_qualified']
 f2=full.copy();f2[0,0]=bypass[0,0];assert not calculate(raw,f2,bypass,y,rx)['source_qualified']
 f2=bypass.copy();ix=np.arange(80);f2[:,:,ix,:]=0;f2[:,:,ix,(y[ix]+1)%6]=30;assert not calculate(raw,f2,bypass,y,rx)['source_qualified']
 for bad in ('shape','nonfinite','RX'):
  r=raw[:,:-1] if bad=='shape' else raw.copy();xx=rx.copy()
  if bad=='nonfinite':r[0,0,0]=np.nan
  if bad=='RX':xx[rx==8]=9
  try:calculate(r,full,bypass,y,xx)
  except ValueError:pass
  else:raise AssertionError('Negative not rejected '+bad)
 m=measure(z,y);assert m['correct']==6300 and m['macro_f1']==1
 print('INDEPENDENT_SYNTHETIC_CHECKS PASS8;0model/IQ/CP/NN')
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def csvfile(p,rows):
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);p.add_argument('--test',action='store_true');a=p.parse_args()
 if a.test:selftest();return
 if a.input is None or a.output is None:raise ValueError('Exact saved input/output needed')
 if a.output.exists():raise FileExistsError('Preserve previous analysis')
 started=time.perf_counter();cfg=json.loads((a.input/'resolved_config.json').read_text());online=json.loads((a.input/'result.json').read_text());done=json.loads((a.input/'completion.json').read_text());c=cfg['config']
 if c['L']!=6300 or c['source_RX']!=list(RX) or c['source_RX_holdout'] is not None or c['U_used'] or c['V_used'] or c['target_access'] or c['training_updates'] or c['head_fit_updates'] or c['expected_NN']!=63004 or done['NN']!=63004 or done['target_access'] or done['status']!='SOURCE_ANALYZED':raise ValueError('Frozen L-only observed execution differs')
 with np.load(a.input/'source_predictions.npz',allow_pickle=False) as f:arr={k:f[k] for k in f.files}
 y=arr['y'];rx=arr['rx'];day=arr['day'];tr=arr['trigger'];mse=arr['mse'];wp=arr['window_points'];raw=arr['raw'];full=arr['full'];bypass=arr['bypass']
 if len(set(arr['ids'].tolist()))!=6300 or set(day.tolist())!={1,2,3} or tr.shape!=(2,6300) or mse.shape!=tr.shape or wp.shape!=tr.shape or not np.isfinite(mse).all() or (mse<0).any():raise ValueError('Saved identity/observation metadata differs')
 if any(int(((y==t)&(rx==r)&(day==d)).sum())!=70 for t in range(6) for r in RX for d in (1,2,3)):raise ValueError('All90 cells required')
 if np.any(mse[~tr]!=0) or np.any(wp[~tr]!=0):raise ValueError('No-trigger physical effect')
 for ai in range(2):
  for dr in range(2):
   if not np.array_equal(full[ai,dr,~tr[dr]],bypass[ai,dr,~tr[dr]]) or not np.array_equal(full[ai,dr,mse[dr]==0],bypass[ai,dr,mse[dr]==0]):raise ValueError('Zero input change requires exact logits')
 r=calculate(raw,full,bypass,y,rx);maximum=0.
 if r['checks']!=online['checks'] or r['cohort_count']!=online['cohort_count']:raise ValueError('Independent decision mismatch')
 for q in r['overall']:
  old=next(x for x in online['overall'] if x['arm']==q['arm'] and x['draw']==q['draw'])
  for key,okey in [('FULL_correct','correct'),('NO_TAPER_correct',None),('FULL_retained','FULL_retained'),('NO_TAPER_retained','NO_TAPER_retained'),('retention_gain_pp','retention_gain_pp'),('accuracy_delta_pp','overall_accuracy_delta_pp'),('repairs','repairs'),('new_errors','new_errors')]:
   val=old['NO_TAPER']['correct'] if okey is None else old[okey];err=abs(float(q[key])-float(val));maximum=max(maximum,err)
   if err>1e-10:raise ValueError('Independent primary mismatch '+key)
 for q,old in zip(r['per_RX'],online['per_RX']):
  if q['arm']!=old['arm'] or q['RX']!=old['RX'] or abs(q['retention_gain_pp']-old['retention_gain_pp'])>1e-10:raise ValueError('RX result mismatch')
 strata=[]
 for ai,arm in enumerate(ARMS):
  for view,dr,z in [('RAW',-1,raw[ai])]+[(v,dr,zz[ai,dr]) for v,zz in [('FULL',full),('NO_TAPER',bypass)] for dr in (0,1)]:
   for dim,groups in [('overall',[('ALL',np.ones(6300,bool))]),('RX',[(str(v),rx==v) for v in RX]),('TX',[(str(v),y==v) for v in range(6)]),('day',[(str(v),day==v) for v in (1,2,3)])]:
    for label,mask in groups:strata.append(dict(arm=arm,view=view,draw=dr,dimension=dim,stratum=label,**measure(z[mask],y[mask])))
 operations=[]
 for dr in (0,1):
  operations.append(dict(draw=dr,observations=6300,native_gate_triggered=int(tr[dr].sum()),IQ_changed=int((mse[dr]>0).sum()),window_points_values={str(v):int((wp[dr]==v).sum()) for v in np.unique(wp[dr])},mean_MSE=float(mse[dr].mean()),max_MSE=float(mse[dr].max()),B00_logits_changed=int(np.any(full[0,dr]!=bypass[0,dr],axis=1).sum()),B01_logits_changed=int(np.any(full[1,dr]!=bypass[1,dr],axis=1).sum())))
 a.output.mkdir();csvfile(a.output/'source_strata.csv',[{k:v for k,v in q.items() if k!='confusion'} for q in strata]);csvfile(a.output/'paired_effects.csv',r['overall']);csvfile(a.output/'source_RX.csv',r['per_RX']);write(a.output/'confusions.json',strata)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained');labs=[q['arm']+'/draw'+str(q['draw']) for q in r['overall']];ax[0].bar(labs,[q['retention_gain_pp'] for q in r['overall']],color=['#456A8F']*2+['#C17828']*2);ax[0].axhline(.2,ls='--',color='red',label='Fixed minimum +0.2 pp');ax[0].axhline(0,color='gray');ax[0].set_ylim(-.27,.3);ax[0].set_ylabel('NO_TAPER - FULL / percentage points');ax[0].set_title('Fixed B00 raw-correct cohort;not training gains');ax[0].tick_params(axis='x',rotation=20);ax[0].legend(fontsize=8)
 for arm,col in zip(ARMS,['#456A8F','#C17828']):ax[1].plot(RX,[q['retention_gain_pp'] for q in r['per_RX'] if q['arm']==arm],'o-',label=arm,color=col)
 ax[1].axhline(0,color='gray');ax[1].axhline(-.25,ls='--',color='red',label='Fixed RX floor');ax[1].set_ylim(-.28,.3);ax[1].set_xticks(RX);ax[1].set_xlabel('All source RX;two ordinary draws pooled');ax[1].set_title('One fixed modelseed;no target observations');ax[1].legend(fontsize=8)
 for ex in ('png','pdf'):fig.savefig(a.output/('source_taper_effects.'+ex),dpi=180)
 plt.close(fig);fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained');xx=np.arange(2);width=.2
 for j,(key,label) in enumerate([('native_gate_triggered','Native taper gate'),('IQ_changed','Nonzero waveform change'),('B00_logits_changed','B00 logits changed'),('B01_logits_changed','B01 logits changed')]):ax[0].bar(xx+(j-1.5)*width,[o[key] for o in operations],width,label=label)
 ax[0].set_xticks(xx,['draw0','draw1']);ax[0].set_ylabel('Observations / 6300 per draw');ax[0].set_title('Native triggering is preserved;no forced gates');ax[0].legend(fontsize=8)
 vals=sorted(set(wp.flatten().tolist()))
 for dr,col in zip((0,1),['#456A8F','#C17828']):ax[1].bar(np.arange(len(vals))+(dr-.5)*.34,[int(((wp[dr]==v)&tr[dr]).sum()) for v in vals],.34,label='draw'+str(dr),color=col)
 ax[1].set_xticks(np.arange(len(vals)),list(map(str,vals)));ax[1].set_xlabel('Original Tukey weights below one (triggered only)');ax[1].set_ylabel('Observations');ax[1].set_title('Actual scheduled window;all256 points retained');ax[1].legend()
 for ex in ('png','pdf'):fig.savefig(a.output/('native_taper_observation.'+ex),dpi=180)
 plt.close(fig)
 result=dict(status='VERIFIED_INDEPENDENT_SOURCE_RECOUNT',source_decision=r,primary_max_error=maximum,logit_values_read=378000,confusion_matrices=len(strata),source_cells=90,source_cell_count=70,operations=operations,actual_E13={k:cfg['actual_E13'][k] for k in ('p_bandedge_taper','taper_alpha_min','taper_alpha_max','p_time_shift','max_time_shift','p_pa')},wall_seconds=time.perf_counter()-started,new_IQ_CP_NN_updates=0,target_access=False,goal_complete=False)
 write(a.output/'independent_recount.json',result);print(json.dumps(result))
if __name__=='__main__':main()
