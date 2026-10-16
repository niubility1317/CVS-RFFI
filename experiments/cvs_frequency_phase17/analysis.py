"""Independent stdlib-only reconstruction of every cell and fixed qualification."""
import cmath,gzip,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'local_artifacts/F_frequency_phase_observability_20261017';OUT=P/'run'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def mean(v):return math.fsum(v)/len(v)
def close(a,b):
 if a is None or b is None:assert a is b
 else:assert math.isclose(a,b,rel_tol=2e-8,abs_tol=2e-10),(a,b)
def reconstruct(real,imag):
 mu=[[[[complex(a,b) for a,b in zip(real[t][d][r],imag[t][d][r])] for r in range(4)] for d in range(3)] for t in range(6)]
 if min(abs(v) for t in mu for d in t for r in d for v in r)<.8:return dict(identifiable=False)
 values=[[[[0.]*8 for r in range(4)] for dd in range(3)] for t in range(6)]
 for t in range(6):
  for dd in range(3):
   for k in range(8):
    phi=[cmath.phase(mu[t][dd][r][k]) for r in range(4)];ref=sum(cmath.exp(1j*q) for q in phi)/4
    if abs(ref)<.8:return dict(identifiable=False)
    ph=[math.atan2(math.sin(q-cmath.phase(ref)),math.cos(q-cmath.phase(ref))) for q in phi];av=mean(ph)
    for r in range(4):values[t][dd][r][k]=ph[r]-av
 profile=[[mean([values[t][dd][r][k] for t in range(6) for dd in range(3)]) for k in range(8)] for r in range(4)]
 folds=[];per_e=[0.]*4;per_z=[0.]*4;wrong=[0.]*3
 for t in range(6):
  for dd in range(3):
   fit=[[mean([values[u][v][r][k] for u in range(6) if u!=t for v in range(3) if v!=dd]) for k in range(8)] for r in range(4)];actual=values[t][dd]
   e=math.fsum((fit[r][k]-actual[r][k])**2 for r in range(4) for k in range(8));z=math.fsum(actual[r][k]**2 for r in range(4) for k in range(8));ww=[math.fsum((fit[(r-shift)%4][k]-actual[r][k])**2 for r in range(4) for k in range(8)) for shift in (1,2,3)]
   for r in range(4):per_e[r]+=math.fsum((fit[r][k]-actual[r][k])**2 for k in range(8));per_z[r]+=math.fsum(actual[r][k]**2 for k in range(8))
   for k in range(3):wrong[k]+=ww[k]
   folds.append(dict(TX=t,day=dd,prediction_SSE=e,zero_SSE=z,wrong_RX_SSE=ww))
 sse=math.fsum(x['prediction_SSE'] for x in folds);zero=math.fsum(x['zero_SSE'] for x in folds);gain=1-sse/zero if zero>1e-18 else None;wg=[1-x/zero if zero>1e-18 else None for x in wrong];margin=gain-max(wg) if gain is not None else None;better=sum(x['prediction_SSE']<x['zero_SSE'] for x in folds);each=all(a<=b+1e-15 for a,b in zip(per_e,per_z));rms=math.sqrt(mean([v*v for row in profile for v in row]));detected=rms>=.02 and gain is not None and gain>=.5 and better>=15 and each and margin>=.05
 return dict(identifiable=True,profile=profile,cell_RX_contrasts=values,profile_RMS_rad=rms,prediction_SSE=sse,zero_SSE=zero,crossfit_risk_improvement=gain,wrong_RX_risk_improvements=wg,correct_minus_best_wrong=margin,better_folds=better,per_RX_prediction_SSE=per_e,per_RX_zero_SSE=per_z,each_RX_not_worse=each,detected_RX=detected,folds=folds)
def compare(a,b):
 if isinstance(a,dict):
  for k,v in a.items():compare(v,b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for u,v in zip(a,b):compare(u,v)
 elif isinstance(a,bool) or isinstance(a,str):assert a==b
 else:close(a,b)
def expected_phase(case):
 if case!='RX_LTI':return [[0.]*8 for _ in range(4)]
 qs=(-6,-5,-4,-3,-2,-1,1,2,3,4,5,6);centers=(-5,-4,-3,-2,2,3,4,5);ph=[]
 for a in (-.45,-.15,.15,.45):
  h={q:(a+cmath.exp(-2j*math.pi*q/20))/(1+a*cmath.exp(-2j*math.pi*q/20)) for q in qs};ph.append([cmath.phase(h[k-1]*h[k+1]*h[k].conjugate()**2) for k in centers])
 for k in range(8):
  av=mean([ph[r][k] for r in range(4)])
  for r in range(4):ph[r][k]-=av
 return ph
def main():
 config=read(OUT/'resolved_config.json');done=read(OUT/'completion.json');assert done['cells']==252 and done['synthetic_packets']==1270080 and done['synthetic_complex_points']==325140480 and done['execution_commit']==config['execution_commit']
 assert not done['real_IQ'] and not done['checkpoint'] and not done['NN'] and not done['updates'] and not done['target_access']
 with gzip.open(OUT/'cells.jsonl.gz','rt',encoding='utf8') as f:rows=[json.loads(v) for v in f if v.strip()]
 assert len(rows)==252;groups={};recount=[];index=0
 for ti,template in enumerate(config['templates']):
  for si,snr in enumerate(config['SNR_dB']):
   for case in config['cases']:
    for draw in range(4):
     r=rows[index];assert r['cell']==index and (r['template'],r['SNR_dB'],r['case'],r['draw'])==(template,snr,case,draw) and r['data_seed']==config['data_seed']+ti*100000+si*1000+draw;index+=1
     e=r['estimate'];counts=e['cell_valid_counts'];assert len(counts)==6 and all(len(a)==3 and all(len(b)==4 for b in a) for a in counts);valid=sum(v for a in counts for b in a for v in b);assert valid==e['valid_packets'] and all(0<=v<=70 for a in counts for b in a for v in b)
     expected=expected_phase(case);compare(expected,r['truth']['expected_RX_phase']);assert r['truth']['synthetic_packets']==5040 and r['truth']['synthetic_complex_points']==1290240
     if valid!=5040:
      assert not e['identifiable'] and e['cellmeans_real'] is None and e['cellmeans_imag'] is None;s=dict(identifiable=False);rel=None;rms=None;passed=case=='DEGENERATE' and 1-valid/5040>=.99
     else:
      s=reconstruct(e['cellmeans_real'],e['cellmeans_imag']);compare(s,{k:e[k] for k in s});rms=s.get('profile_RMS_rad');rel=None
      if not s['identifiable']:passed=False
      elif case=='RX_LTI':
       rel=math.sqrt(math.fsum((s['profile'][i][j]-expected[i][j])**2 for i in range(4) for j in range(8))/math.fsum(v*v for row in expected for v in row));passed=s['detected_RX'] and rel<=.25
      elif case=='TX_RX_INTERACTION':passed=(not s['detected_RX']) and s['crossfit_risk_improvement'] is not None and s['crossfit_risk_improvement']<=0 and rms<=.01
      elif case=='DEGENERATE':passed=False
      else:passed=(not s['detected_RX']) and rms<=.01
     close(rel,r['metrics']['profile_relative_RMSE']);close(1-valid/5040,r['metrics']['rejected_fraction']);assert bool(passed)==r['metrics']['passed'];key=(template,case,snr);groups.setdefault(key,[]).append(dict(draw=draw,passed=bool(passed),identifiable=bool(s['identifiable']),valid_packets=valid,profile_RMS_rad=rms,profile_relative_RMSE=rel,risk_improvement=s.get('crossfit_risk_improvement'),better_folds=s.get('better_folds'),detected_RX=s.get('detected_RX',False)))
     recount.append(dict(cell=r['cell'],template=template,case=case,SNR_dB=snr,draw=draw,passed=bool(passed)))
 summaries=[dict(template=t,case=c,SNR_dB=n,passing_draws=sum(x['passed'] for x in v),draws=len(v),passed=all(x['passed'] for x in v),results=v) for (t,c,n),v in groups.items()];assert len(summaries)==63 and all(s['draws']==4 for s in summaries)
 out=dict(status='ANALYZED',actual_execution='VERIFIED',run_id=config['run_id'],executed_code_commit=done['execution_commit'],qualified=all(s['passed'] for s in summaries),passed_groups=sum(s['passed'] for s in summaries),groups=63,passed_cells=sum(r['passed'] for r in recount),cells=252,group_results=summaries,all_cell_reconstructions_agree=True,truth_used_only_for_scoring=True,real_IQ=0,checkpoint=0,NN=0,updates=0,teacher=0,head_fit=0,target_access=False,synthetic_packets=done['synthetic_packets'],synthetic_complex_points=done['synthetic_complex_points'],wall_seconds=done['wall_seconds'],Ag_efficacy_proved=False,joint_goal_complete=False)
 (P/'analysis_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps({k:out[k] for k in ('status','qualified','passed_groups','groups','passed_cells','cells','synthetic_packets','synthetic_complex_points','wall_seconds')}))
if __name__=='__main__':main()
