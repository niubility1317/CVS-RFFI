import numpy as np
from . import design as d
def metrics(z,y,mask):
 p=z.argmax(-1);q=np.asarray(mask,bool);n=int(q.sum())
 if not n:raise ValueError('Empty fixed source stratum')
 other=z.copy();other[np.arange(len(y)),y]=-np.inf;m=(z[np.arange(len(y)),y]-other.max(-1))/30
 return dict(count=n,correct=int((p[q]==y[q]).sum()),accuracy=float((p[q]==y[q]).mean()),mean_margin_over30=float(m[q].mean()))
def summarize(raw,full,bypass,y,rx,trigger):
 if raw.shape!=(2,6300,6) or full.shape!=(2,2,6300,6) or bypass.shape!=full.shape or y.shape!=(6300,) or set(rx.tolist())!=set(d.RX) or trigger.shape!=(2,6300):raise ValueError('Exact frozen full source shapes required')
 if not all(np.isfinite(v).all() for v in (raw,full,bypass)):raise ValueError('Nonfinite logits')
 cohort=raw[0].argmax(-1)==y;rows=[];perRX=[];gains=[]
 for ai,a in enumerate(d.ARMS):
  for draw in range(2):
   pf=full[ai,draw].argmax(-1)==y;pb=bypass[ai,draw].argmax(-1)==y
   if not np.array_equal(full[ai,draw,~trigger[draw]],bypass[ai,draw,~trigger[draw]]):raise ValueError('Nontrigger logit equality failed')
   for dim,groups in [('overall',[('ALL',np.ones(6300,bool))]),('RX',[(str(r),rx==r) for r in d.RX])]:
    for label,mask in groups:
     cm=mask&cohort;f=metrics(full[ai,draw],y,mask);b=metrics(bypass[ai,draw],y,mask);gn=float(100*(pb[cm].mean()-pf[cm].mean()))
     rec=dict(arm=a,draw=draw,dimension=dim,stratum=label,**f,NO_TAPER=b,fixed_B00_raw_correct_count=int(cm.sum()),FULL_retained=int(pf[cm].sum()),NO_TAPER_retained=int(pb[cm].sum()),retention_gain_pp=gn,overall_accuracy_delta_pp=100*(b['accuracy']-f['accuracy']),repairs=int((pb&~pf&mask).sum()),new_errors=int((pf&~pb&mask).sum()),triggered_observations=int((trigger[draw]&mask).sum()))
     rows.append(rec)
     if dim=='overall':gains.append(rec)
  for r in d.RX:
   cm=(rx==r)&cohort;pf=full[ai].argmax(-1)==y[None,:];pb=bypass[ai].argmax(-1)==y[None,:];perRX.append(dict(arm=a,RX=r,count_per_draw=int(cm.sum()),retention_gain_pp=float(100*(pb[:,cm].mean()-pf[:,cm].mean()))))
 checks=dict(each_model_draw_retention=all(q['retention_gain_pp']>=.2-1e-10 for q in gains),each_model_draw_overall_nonnegative=all(q['overall_accuracy_delta_pp']>=-1e-10 for q in gains),each_model_at_least4positive_RX=all(sum(v['retention_gain_pp']>0 for v in perRX if v['arm']==a)>=4 for a in d.ARMS),each_model_worst_RX_preserved=all(v['retention_gain_pp']>=-.25-1e-10 for v in perRX))
 return dict(status='SOURCE_ANALYZED',source_qualified=all(checks.values()),decision='PASS_FIXED_TAPER_INPUT_PREMISE_ONLY' if all(checks.values()) else 'FAIL_FIXED_TAPER_INPUT_PREMISE_CLOSED',checks=checks,overall=gains,per_RX=perRX,strata=rows,cohort_count=int(cohort.sum()),raw=[metrics(raw[i],y,np.ones(6300,bool)) for i in range(2)],unique_source_L=6300,model_seed_count=1,channel_draws=0,ordinary_draws=2,new_NN=63004,new_training_updates=0,Ag_proved=False,training_gain_proved=False,automatic_training=False,target_access=False,goal_complete=False)
