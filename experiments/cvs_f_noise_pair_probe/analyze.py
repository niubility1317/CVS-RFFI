"""Independent offline recount from small full derivative Gram records and decisions."""
import argparse,gzip,json,math
from collections import Counter,defaultdict
from pathlib import Path
from . import design as d
from .core import GROUPS,aggregate

def near(a,b):
 if not math.isfinite(a+b) or abs(a-b)>1e-8*max(1.,abs(a),abs(b)):raise ValueError('Independent scalar mismatch')
def from_gram(row):
 K=d.DRAWS;G=row['raw_derivative_gram']
 if len(G)!=2*K or any(len(v)!=2*K for v in G) or any(not math.isfinite(x) for v in G for x in v):raise ValueError('Full finite derivative Gram required')
 for i in range(2*K):
  if G[i][i]<-1e-10:raise ValueError('Negative squared norm')
  for j in range(i):near(G[i][j],G[j][i])
 def variance(a,b):return (sum(G[i+a][i+b] for i in range(K))-sum(G[i+a][j+b] for i in range(K) for j in range(K))/K)/(K-1)
 vp=variance(0,0);vm=variance(K,K);cov=variance(0,K);vi=(vp+vm)/4;va=vi+cov/2
 if min(vp,vm,vi,va)<-1e-8*max(1.,vp,vm):raise ValueError('Invalid variance')
 for key,value in dict(Vplus=vp,Vminus=vm,noise_covariance=cov,independent_pair_trace=vi,antithetic_pair_trace=va).items():near(row[key],value)
 return dict(group=row['group'],batch=row['batch'],count=row['count'],Vplus=vp,Vminus=vm,noise_covariance=cov,independent_pair_trace=vi,antithetic_pair_trace=va,trace_ratio=va/vi if vi>0 else None)
def score(preds,labels):
 C=[[0]*6 for _ in range(6)]
 for pid,pred in preds:C[labels[pid]['TX']][pred]+=1
 n=sum(map(sum,C));correct=sum(C[k][k] for k in range(6));F=[]
 for k in range(6):
  den=sum(C[k])+sum(row[k] for row in C);F.append(2*C[k][k]/den if den else 0.)
 return dict(count=n,accuracy=correct/n,macro_F1=sum(F)/6,confusion=C)
def analyze(root):
 root=Path(root);complete=d.read(root/'completion.json')
 if complete['status']!='ARTIFACTS_COMPLETE' or (root/'failure.json').exists():raise ValueError('Complete fixed prediction first')
 summaries={};common_observations=None;all_gradient=[];strata=[]
 for row in d.rows():
  out=root/row['row_id'];config=d.read(out/'resolved_config.json');d.validate(config);actual=d.read(out/'summary.json');smoke=d.read(out/'checkpoint_noquery_smoke.json');labels={v['physical_id']:v for v in d.read(out/'source_labels.json')};contract=d.read(out/'input_contract.json')
  if len(labels)!=5040 or set(labels)!=set(contract['role_ids']['L_s']) or any(x['RX'] not in d.FIT or x['day'] not in (1,2,3) or x['TX'] not in range(6) for x in labels.values()):raise ValueError('Complete legal fitL labels required')
  cells=Counter((x['TX'],x['RX'],x['day']) for x in labels.values())
  if len(cells)!=72 or set(cells.values())!={70}:raise ValueError('Full source physical cells required')
  if actual['NN']!=config['expected_NN'] or actual['decisions']!=config['expected_decisions'] or not actual['parameters_and_buffers_unchanged'] or any(actual[k] for k in ('optimizer_updates','head_fit_updates','EMA_updates','model_epochs','target_access')) or smoke['synthetic_NN_packets']!=2 or smoke['IQ_before_smoke']:raise ValueError('Actual zero-update/default-head budget differs')
  observed={};batch_ids={};obs_lines=[]
  with gzip.open(out/'noise_observations.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    x=json.loads(line);obs_lines.append(x);batch=x['batch'];ids=x['ids']
    if batch in batch_ids or len(ids)!=len(set(ids)) or set(ids)-set(labels) or len(ids)!=(48 if batch==39 else 128):raise ValueError('Bad source physical batch')
    batch_ids[batch]=ids
    for p in x['records']:
     key=(batch,p['draw'],p['sign'],p['sample_id'])
     if key in observed or p['sample_id'] not in ids or p['draw'] not in range(d.DRAWS) or p['sign'] not in (1,-1) or p['RX']!=labels[p['sample_id']]['RX'] or p['day']!=labels[p['sample_id']]['day']:raise ValueError('Incomplete/mispaired channel observation')
     observed[key]=p
  if len(observed)!=config['expected_decisions'] or sorted(batch_ids)!=list(range(40)) or len(set(pid for ids in batch_ids.values() for pid in ids))!=5040:raise ValueError('Full40batch/5040physical coverage required')
  if common_observations is None:common_observations=obs_lines
  elif common_observations!=obs_lines:raise ValueError('Two seed physical/noise observations differ')
  with gzip.open(out/'gradient_statistics.jsonl.gz','rt',encoding='utf8') as f:records=[json.loads(v) for v in f]
  mapped={(v['batch'],v['group']):v for v in records}
  if len(records)!=240 or len(mapped)!=240 or set(mapped)!={(b,g) for b in range(40) for g in (*GROUPS,'whole')}:raise ValueError('Full derivative groups required')
  if any(v['count']!=len(batch_ids[v['batch']]) or v['draws']!=d.DRAWS for v in records):raise ValueError('Gradient batch/count/draw contract differs')
  recounted=[]
  for b in range(40):
   whole=mapped[b,'whole']['raw_derivative_gram']
   for i in range(2*d.DRAWS):
    for j in range(2*d.DRAWS):near(whole[i][j],sum(mapped[b,g]['raw_derivative_gram'][i][j] for g in GROUPS))
   for g in (*GROUPS,'whole'):
    v=from_gram(mapped[b,g]);v['model_seed']=row['model_seed'];recounted.append(v)
  decision=aggregate([v for v in recounted if v['group']=='whole'])
  for k in ('trace_ratio','negative_covariance_fraction'):near(decision[k],actual['diagnostic'][k])
  if decision['qualified_rationale']!=actual['diagnostic']['qualified_rationale']:raise ValueError('Independent decision differs')
  predictions={};by_pair=defaultdict(list);by_stratum=defaultdict(list)
  with gzip.open(out/'predictions.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    p=json.loads(line);key=(p['batch'],p['draw'],p['sign'],p['physical_id']);t=labels.get(p['physical_id'])
    if key not in observed or key in predictions or t is None or any(t[k]!=p[k] for k in ('TX','RX','day')) or p['prediction'] not in range(6) or not math.isfinite(p['margin_over30']+p['CE_batch']):raise ValueError('Prediction roles/count/margin differ')
    predictions[key]=p;entry=(p['physical_id'],p['prediction']);by_pair[p['draw'],p['sign']].append(entry);phys=observed[key]
    for dim in ('scene','state','lock','RX','day'):by_stratum[dim,str(phys[dim])].append(entry)
  if set(predictions)!=set(observed):raise ValueError('Complete fitL predictions required')
  scores={f'draw{k}_sign{s}':score(v,labels) for (k,s),v in sorted(by_pair.items())}
  for (dimension,value),v in sorted(by_stratum.items()):strata.append(dict(model_seed=row['model_seed'],dimension=dimension,value=value,**score(v,labels)))
  summaries[str(row['model_seed'])]=dict(diagnostic=decision,scores=scores,wall_seconds=actual['wall_seconds'],parent_epochs=20,parent_updates=4440,parent_NN=actual['parent_NN'],NN=actual['NN'],decisions=len(predictions),parameters_and_buffers_unchanged=True,head_fit_updates=0)
  all_gradient.extend(recounted)
 result=dict(status='ANALYZED',actual_execution='VERIFIED',source_only=True,target_access=False,run_id=d.RUN,rows=summaries,qualified_variance_rationale=all(v['diagnostic']['qualified_rationale'] for v in summaries.values()),same_observation_records_across_model_seeds=True,new_NN=sum(v['NN'] for v in summaries.values()),optimizer_updates=0,head_fit_updates=0,EMA_updates=0,model_epochs=0,performance_improvement_proved=False,Ag_efficacy_proved=False,joint_goal_complete=False,on_fail='Close fixed variance rationale,no layer/state/RX/seed/draw/threshold rescue',limits=['Frozen eval-mode CE derivatives only,not full training/RDrop KL/SupCon/AdamW trajectory','Independent2noise reference estimated from unbiased marginal sample variances;no separately realized independent pair forwards','8draws within each fixed physical channel,40fitbatches;no heldRX or target inference','Repeatedviews of5040existing packets are not new independent physical observations'])
 d.write(root/'analysis_summary.json',result);d.write(root/'independent_gradient_records.json',all_gradient);d.write(root/'source_strata.json',strata);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();print(json.dumps(analyze(a.input)))
