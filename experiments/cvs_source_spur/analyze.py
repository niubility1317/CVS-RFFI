"""Independent Python-loop recount of saved aggregate PSD;zero IQ/model access."""
import argparse,json,math,statistics
from collections import Counter
from pathlib import Path
import numpy as np

def recount(root):
 root=Path(root);row=root/'nonDC-line-fitRX3_4_6_8';s=json.loads((row/'summary.json').read_text());c=json.loads((row/'resolved_config.json').read_text());saved=np.load(row/'source_line_statistics.npz',allow_pickle=False);raw=saved['cell_PSD'];recorded=saved['cell_prominence_db'];bins=saved['shifted_bins'].tolist()
 expected=[k for k in range(8,248) if abs(k-128)>8]
 if raw.shape!=(2,6,3,4,256) or recorded.shape!=(2,6,3,4,223) or bins!=expected or not np.isfinite(raw).all() or np.any(raw<0):raise ValueError('Complete finite registered aggregation required')
 if any(abs(sum(p)/256-1)>1e-12 for p in raw.reshape(-1,256).tolist()):raise ValueError('Normalized full-cell PSD inconsistent')
 p=raw.tolist();a=[[[[[0. for _ in bins] for r in range(4)] for day in range(3)] for t in range(6)] for w in range(2)]
 for w in range(2):
  for t in range(6):
   for day in range(3):
    for r in range(4):
     v=p[w][t][day][r]
     for b,k in enumerate(bins):a[w][t][day][r][b]=10*math.log10(max(sum(v[k+j] for j in (-1,0,1))/3,1e-12)/max(statistics.median(v[k+j] for j in (-7,-6,-5,-4,4,5,6,7)),1e-12))
 maxerr=max(abs(a[w][t][dd][r][b]-float(recorded[w,t,dd,r,b])) for w in range(2) for t in range(6) for dd in range(3) for r in range(4) for b in range(223))
 if maxerr>1e-10:raise ValueError('Saved prominence differs')
 folds=[];errors=[[] for _ in range(2)];nulls=[[] for _ in range(2)];wrongs=[[] for _ in range(2)];sel=[];success=0
 for t in range(6):
  for day in range(3):
   fitted=[[[sum(a[w][tt][dd][r][b] for tt in range(6) if tt!=t for dd in range(3) if dd!=day)/10 for b in range(223)] for r in range(4)] for w in range(2)]
   candidates=[(fitted[0][r][b]-sum(fitted[0][rr][b] for rr in range(4))/4,r,b) for r in range(4) for b in range(223)]
   # First maximum reproduces predeclared lexicographic tie handling.
   best=max(x[0] for x in candidates);_,r,b=next(x for x in candidates if x[0]==best);sel.append((r,bins[b]));fp=[];fc=[];hp=[];hc=[];se=[];zz=[];ww=[]
   for w in range(2):
    pred=[fitted[w][rr][b]-sum(fitted[w][r0][b] for r0 in range(4))/4 for rr in range(4)];obs=[a[w][t][day][rr][b]-sum(a[w][t][day][r0][b] for r0 in range(4))/4 for rr in range(4)]
    e=[(u-v)**2 for u,v in zip(pred,obs)];z=[v*v for v in obs];wrong=[sum((pred[(rr-k)%4]-obs[rr])**2 for rr in range(4)) for k in (1,2,3)];errors[w].append(e);nulls[w].append(z);wrongs[w].append(wrong);fp.append(fitted[w][r][b]);fc.append(pred[r]);hp.append(a[w][t][day][r][b]);hc.append(obs[r]);se.append(sum(e));zz.append(sum(z));ww.append(wrong)
   ok=all(v>=6 for v in fp) and all(v>=3 for v in fc+hp+hc);success+=ok
   folds.append(dict(TX_index=t,day=day+1,selected_RX=(3,4,6,8)[r],selected_shifted_bin=bins[b],selected_frequency_hz=(bins[b]-128)*25000000/256,fit_prominence_db=fp,fit_RX_contrast_db=fc,held_prominence_db=hp,held_RX_contrast_db=hc,line_success=ok,SSE=se,zero_SSE=zz,wrong_RX_SSE=ww))
 def near(x,y):
  if isinstance(x,dict):return set(x)==set(y) and all(near(x[k],y[k]) for k in x)
  if isinstance(x,list):return len(x)==len(y) and all(near(u,v) for u,v in zip(x,y))
  if x is None or isinstance(x,(str,bool)):return x==y
  return math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-10)
 if not near(folds,s['folds']):raise ValueError('At least one full source fold differs')
 windows=[]
 for w,name in enumerate(('rectangular','hann')):
  e,z=errors[w],nulls[w];num=sum(map(sum,e));den=sum(map(sum,z));gain=lambda nums:None if sum(z[i][r] for i,r in nums)<=1e-20 else 1-sum(e[i][r] for i,r in nums)/sum(z[i][r] for i,r in nums)
  wrong=[sum(v[k] for v in wrongs[w]) for k in range(3)]
  windows.append(dict(window=name,risk_reduction=None if den<=1e-20 else 1-num/den,positive_folds=sum(sum(ee)<sum(zz) for ee,zz in zip(e,z)),correct_over_best_wrong_RX=None if den<=1e-20 else (min(wrong)-num)/den,TX_risk_reductions=[gain([(i,r) for i in range(3*t,3*t+3) for r in range(4)]) for t in range(6)],day_risk_reductions=[gain([(i,r) for i in range(dd,18,3) for r in range(4)]) for dd in range(3)],RX_risk_reductions=[gain([(i,r) for i in range(18)]) for r in range(4)],SSE=num,zero_SSE=den,wrong_RX_SSE=wrong))
 if not near(windows,s['windows']):raise ValueError('Window risk aggregate differs')
 mode,count=Counter(sel).most_common(1)[0];checks=dict(complete_finite_grid=True,successful_folds=success>=15,same_RX_bin=count>=15,risk=all(v['risk_reduction'] is not None and v['risk_reduction']>=.5 for v in windows),positive_folds=all(v['positive_folds']>=15 for v in windows),wrong_RX_controls=all(v['correct_over_best_wrong_RX'] is not None and v['correct_over_best_wrong_RX']>=.2 for v in windows),all_TX_day_RX=all(x is not None and x>0 for v in windows for key in ('TX_risk_reductions','day_risk_reductions','RX_risk_reductions') for x in v[key]))
 if checks!=s['checks'] or success!=s['successful_folds'] or count!=s['modal_count'] or ('PASS_SOURCE_ASSOCIATION_ONLY' if all(checks.values()) else 'FAIL')!=s['status']:raise ValueError('Scientific decision differs')
 completion=json.loads((root/'completion.json').read_text());log=(root/'dispatcher.log').read_text();execution=json.loads((root/'execution.json').read_text());pre=json.loads((root/'preflight.json').read_text());submit=json.loads((root/'submit.json').read_text())
 if completion['status']!='ARTIFACTS_COMPLETE' or (root/'failure.json').exists() or 'COMPLETE ' not in log or len(log.splitlines())!=1 or any(s.get(k)!=0 for k in ('model_forwards','checkpoint_loads','optimizer_updates')) or s['physical_packet_counts']!={'L':5040,'U':0,'V':0,'held_RX1':0,'target':0}:raise ValueError('Complete zero-model source execution required')
 if execution['actual_code_commit']!=pre['commit'] or pre['commit']!=submit['commit'] or not submit['independent_proc_readback']['CUDA_hidden']:raise ValueError('Execution identity differs')
 out=dict(execution='VERIFIED',status='ANALYZED',scientific_status=s['status'],checks=checks,successful_folds=success,modal_RX=(3,4,6,8)[mode[0]],modal_shifted_bin=mode[1],modal_count=count,all_selected_RX_bin_counts=[dict(RX=(3,4,6,8)[r],shifted_bin=b,count=n) for (r,b),n in Counter(sel).most_common()],windows=windows,folds=folds,prominence_values_recounted=32112,PSD_cells_recounted=144,max_saved_prominence_error=maxerr,full_stdout_lines=len(log.splitlines()),CPU_wall_seconds=s['wall_seconds'],model_forwards=0,optimizer_updates=0,checkpoint_loads=0,offline_IQ_reads=0,source_only=True,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,code_commit=pre['commit'])
 (root/'independent_analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8');return out
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);args=parser.parse_args();result=recount(args.input);print(json.dumps({k:result[k] for k in ('execution','scientific_status','successful_folds','modal_count','max_saved_prominence_error')}))
