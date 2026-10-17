"""Independent saved-factor recount. Never opens source IQ, model or raw truth."""
import argparse,json,time
from pathlib import Path
import numpy as np
RX=(1,3,4,6,8)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def run(path):
 start=time.perf_counter();base=Path(path);done=read(base/'completion.json');result=read(base/'result.json');meta=read(base/'source_ids.json');resolved=read(base/'resolved_config.json');config=resolved['config']
 if done['status']!='ARTIFACTS_COMPLETE_SOURCE_ONLY' or done['L']!=6300 or done['V']!=27000 or done['source_RX']!=list(RX) or done['source_RX_holdout'] is not None or any(done[k] for k in ('CP','NN','teacher_NN','updates','head_fit_updates','U_used','U_truth_read','target_access','automatic_training','IQ_views_generated')) or done['IQ_modified_for_training'] or config['window']!=[80,160] or config['period']!=20 or config['relative_epsilon']!=1e-12 or not resolved['source_roles_exact_match']:raise ValueError('Fixed scope/data/model/role contract differs')
 means={};errors=[];ids={};recounted=0
 with np.load(base/'observations.npz',allow_pickle=False) as obs,np.load(base/'cell_matrices.npz',allow_pickle=False) as cells:
  expected={role+'_'+key for role in ('L_s','V') for key in ('factor_real','factor_imag','lag20_coherence','coherent_fraction','window_energy','zero_mean')}
  if set(obs.files)!=expected or set(cells.files)!={'L_s','V'}:raise ValueError('Unexpected saved role/statistic arrays')
  for role,n,k in [('L_s',6300,70),('V',27000,300)]:
   md=meta[role];ids[role]={tuple(v) if isinstance(v,list) else v for v in md['ids']}
   if len(md['ids'])!=n or len(ids[role])!=n or len(md['y'])!=n or len(md['rx'])!=n or len(md['day'])!=n:raise ValueError('Complete original unique source metadata required')
   a,b=obs[role+'_factor_real'],obs[role+'_factor_imag'];u=a+1j*b
   if u.shape!=(n,20) or a.dtype!=np.float64 or b.dtype!=np.float64 or not np.isfinite(u).all() or np.any((abs(u)**2).sum(1)>1+1e-12):raise ValueError('Bounded saved20complex factors required')
   for key in ('lag20_coherence','coherent_fraction','window_energy','zero_mean'):
    q=obs[role+'_'+key]
    if q.shape!=(n,) or not np.isfinite(q).all() or np.any(q<0) or key in ('lag20_coherence','coherent_fraction') and np.any(q>1+1e-12):raise ValueError('Complete bounded source quality required')
   means[role]=np.empty((5,3,6,20,20),complex)
   for i,r in enumerate(RX):
    for j,d in enumerate((1,2,3)):
     for t in range(6):
      select=np.array([yy==t and rr==r and dd==d for yy,rr,dd in zip(md['y'],md['rx'],md['day'])]);v=u[select]
      if len(v)!=k:raise ValueError('Every original source cell must retain all70L/300V')
      m=v.T@v.conj()/k;means[role][i,j,t]=m;errors.append(float(np.max(abs(m-cells[role][i,j,t]))));saved=next(z for z in result['cells'][role] if (z['RX'],z['day'],z['TX'])==(r,d,t));trace=float(np.trace(m).real);norm=float(np.linalg.norm(m));tr=(abs(v)**2).sum(1);disp=max(0.,float(np.mean(tr**2)-norm**2))
      if saved['count']!=k or any(abs(saved[key]-value)>1e-12 for key,value in [('mean_trace',trace),('Frobenius_norm',norm),('projector_population_Frobenius_dispersion',disp)]) or np.min(np.linalg.eigvalsh(m))< -1e-12:raise ValueError('Source cell PSD/reduction recount differs')
   recounted+=n
   if len(result['cells'][role])!=90 or result['quality'][role]['packets']!=n or not result['quality'][role]['all_packets_retained'] or result['quality'][role]['zero_mean_packets']!=int(obs[role+'_zero_mean'].sum()):raise ValueError('Source cell/quality retention differs')
   for key,field in [('lag20_coherence','lag20_coherence_quantiles'),('coherent_fraction','coherent_fraction_quantiles')]:
    if np.max(abs(np.quantile(obs[role+'_'+key],[0,.1,.5,.9,1])-result['quality'][role][field]))>1e-12:raise ValueError('Complete source quality quantiles differ')
 if ids['L_s']&ids['V'] or max(errors)>1e-12:raise ValueError('Role overlap or complete matrix mismatch')
 L,V=means['L_s'],means['V'];B=np.zeros((18,5));E=B.copy();C=B.copy();fold_index=0;maxmetric=0.
 for day in range(3):
  for tx in range(6):
   pred=np.zeros((5,20,20),complex);txpred=pred.copy()
   for dd in range(3):
    if dd==day:continue
    txpred+=(L[:,dd,tx]-L[:,dd,tx].mean(0,keepdims=True))/2
    for tt in range(6):
     if tt!=tx:pred+=(L[:,dd,tt]-L[:,dd,tt].mean(0,keepdims=True))/10
   actual=V[:,day,tx]-V[:,day,tx].mean(0,keepdims=True)
   for i in range(5):B[fold_index,i]=float(np.vdot(actual[i],actual[i]).real);E[fold_index,i]=float(np.vdot(actual[i]-pred[i],actual[i]-pred[i]).real);C[fold_index,i]=float(np.vdot(actual[i]-txpred[i],actual[i]-txpred[i]).real)
   f=result['folds'][fold_index]
   if (f['TX'],f['day'],f['fit_L_cells'],f['evaluation_V_cells'])!=(tx,day+1,50,5):raise ValueError('Fixed statistical fold differs')
   for key,value in [('baseline_SSE',B[fold_index].sum()),('shared_RX_SSE',E[fold_index].sum()),('TXconditional_SSE',C[fold_index].sum())]:maxmetric=max(maxmetric,abs(f[key]-value))
   fold_index+=1
 bs,es,cs=(float(a.sum()) for a in (B,E,C));gain=1-es/bs if bs>1e-20 else None;positive=int(sum(bb.sum()>1e-20 and ee.sum()<bb.sum() for bb,ee in zip(B,E)));rms=float(np.sqrt(B.mean()));checks=dict(nontrivial_RX_shape=rms>=.02,relative_risk_gain=gain is not None and gain>=.20,positive_folds=positive>=15,each_RX_nonnegative=bool(np.all(E.sum(0)<=B.sum(0)+1e-20)),not_strong_TX_dependence=es<=1.05*cs+1e-20)
 for key,value in [('baseline_SSE',bs),('shared_RX_SSE',es),('TXconditional_SSE',cs),('baseline_Frobenius_RMS',rms),('risk_gain',gain)]:
  if value is None:
   if result[key] is not None:raise ValueError('Null zero-contrast metric differs')
  else:maxmetric=max(maxmetric,abs(result[key]-value))
 if maxmetric>1e-11 or positive!=result['positive_folds'] or checks!=result['conditions'] or (result['decision']=='SOURCE_PERIODIC_SHAPE_ASSOCIATION_ONLY')!=all(checks.values()):raise ValueError('Independent source decision or metric differs')
 for i,r in enumerate(RX):
  for key,value in [('baseline_SSE',float(B[:,i].sum())),('shared_RX_SSE',float(E[:,i].sum()))]:
   if abs(result['per_RX'][str(r)][key]-value)>1e-11:raise ValueError('RX partition recount differs')
 return dict(status='VERIFIED',run_id=result['run_id'],decision=result['decision'],conditions=checks,source_L=6300,source_V=27000,all_source_RX=list(RX),source_RX_holdout=None,source_complex_factor_entries_recounted=recounted*20,source_cell_matrix_entries_recounted=72000,source_cells=180,statistical_folds=18,mean_matrix_max_error=max(errors),metric_max_error=maxmetric,IQ_reads=0,CP=0,NN=0,updates=0,U_truth_reads=0,raw_target_truth_reads=0,target_access=False,IQ_operator_implemented=False,joint_goal_complete=False,wall_seconds=time.perf_counter()-start)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();print(json.dumps(run(a.run),allow_nan=False))
