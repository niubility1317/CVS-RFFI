"""Offline full-logit source-guard recount. No IQ/checkpoint/model access."""
import argparse,collections,gzip,json,math,statistics,time
from pathlib import Path
SEEDS=(2026092701,2026092702)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def score(y,p):
 cm=[[0]*6 for _ in range(6)]
 for a,b in zip(y,p):cm[a][b]+=1
 return dict(count=len(y),accuracy=sum(cm[i][i] for i in range(6))/len(y),macro_F1=sum(2*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)) if sum(cm[i])+sum(row[i] for row in cm) else 0 for i in range(6))/6,confusion=cm)
def summary(v):
 z=sorted(v)
 def q(t):
  a=(len(z)-1)*t;i=int(a);return z[i]+(z[min(i+1,len(z)-1)]-z[i])*(a-i)
 return dict(count=len(z),mean=statistics.mean(z),minimum=z[0],p10=q(.1),median=statistics.median(z),p90=q(.9),maximum=z[-1])
def run(root,out):
 start=time.perf_counter();models={};same=[]
 for seed in SEEDS:
  base=root/'guard'/('s'+str(seed));meta=read(base/'source_metadata.json');expected=read(base/'result.json');y=meta['labels'];rx=meta['RX'];day=meta['day'];ids=meta['physical_ids']
  assert len(y)==len(rx)==len(day)==len(ids)==5040 and len({json.dumps(i,sort_keys=True) for i in ids})==5040
  cells=collections.Counter(zip(y,rx,day));assert len(cells)==72 and set(cells.values())=={70} and set(rx)=={3,4,6,8}
  pred={};NN=0
  with gzip.open(base/'predictions.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    r=json.loads(line);key=(r['view'],r['draw']);a=r['anchor'];z=r['logits'];assert 0<=a<5040 and len(z)==6 and all(math.isfinite(v) for v in z)
    group=pred.setdefault(key,{});assert a not in group;group[a]=z;NN+=1
  keys={('raw',0),*((v,d) for v in ('MASK','GAIN') for d in range(1,5))};assert set(pred)==keys and NN==45360 and all(set(v)==set(range(5040)) for v in pred.values())
  obs={}
  with gzip.open(base/'observations.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    r=json.loads(line);n=len(r['anchors'])
    fields={k:v for k,v in r.items() if k not in ('draw','anchors')};assert all(len(v)==n for v in fields.values())
    for j,a in enumerate(r['anchors']):
     key=(r['draw'],a);assert key not in obs;obs[key]={k:v[j] for k,v in fields.items()}
  assert len(obs)==20160 and set(obs)=={(d,a) for d in range(1,5) for a in range(5040)}
  raw=pred['raw',0];rawp={a:max(range(6),key=raw[a].__getitem__) for a in raw}
  def table(v,d,subset=None):
   anchors=range(5040) if subset is None else subset;records=[(a,pred[v,d][a]) for a in anchors];truth=[y[a] for a,z in records];p=[max(range(6),key=z.__getitem__) for a,z in records];r=score(truth,p)
   raw_correct=[i for i,(a,z) in enumerate(records) if rawp[a]==y[a]];r['raw_correct_retention']=sum(p[i]==truth[i] for i in raw_correct)/len(raw_correct) if raw_correct else None
   r['lost']=sum(rawp[a]==y[a] and b!=y[a] for (a,z),b in zip(records,p));r['recovered']=sum(rawp[a]!=y[a] and b==y[a] for (a,z),b in zip(records,p));r['disagreement']=sum(rawp[a]!=b for (a,z),b in zip(records,p))
   r['margin_over30']=summary([(z[y[a]]-max(t for c,t in enumerate(z) if c!=y[a]))/30 for a,z in records]);r['margin_change_over30']=summary([((z[y[a]]-max(t for c,t in enumerate(z) if c!=y[a]))-(raw[a][y[a]]-max(t for c,t in enumerate(raw[a]) if c!=y[a])))/30 for a,z in records]);return r
  results={}
  for v,d in sorted(keys):
   q=table(v,d);q['by_RX']={str(k):table(v,d,[i for i in range(5040) if rx[i]==k]) for k in (3,4,6,8)};q['by_TX']={str(k):table(v,d,[i for i in range(5040) if y[i]==k]) for k in range(6)};q['by_day']={str(k):table(v,d,[i for i in range(5040) if day[i]==k]) for k in (1,2,3)};results[v+str(d)]=q
  def pooled(v,subset=lambda d,a:True):
   truth=[];pp=[];rawc=lost=recovered=disagree=0;mm=[];dm=[]
   for d in range(1,5):
    for a,z in pred[v,d].items():
     if not subset(d,a):continue
     t=y[a];b=max(range(6),key=z.__getitem__);truth.append(t);pp.append(b);was=rawp[a]==t;rawc+=was;lost+=was and b!=t;recovered+=not was and b==t;disagree+=rawp[a]!=b;mr=(z[t]-max(t2 for c,t2 in enumerate(z) if c!=t))/30;mm.append(mr);dm.append(mr-(raw[a][t]-max(t2 for c,t2 in enumerate(raw[a]) if c!=t))/30)
   r=score(truth,pp);r.update(raw_correct_retention=(rawc-lost)/rawc,lost=lost,recovered=recovered,disagreement=disagree,margin_over30=summary(mm),margin_change_over30=summary(dm));return r
  base_score=results['raw0'];identity_checks=[]
  for v in ('MASK','GAIN'):
   p=pooled(v);p['by_RX']={str(k):pooled(v,lambda d,a:rx[a]==k) for k in (3,4,6,8)};p['by_TX']={str(k):pooled(v,lambda d,a:y[a]==k) for k in range(6)};p['by_day']={str(k):pooled(v,lambda d,a:day[a]==k) for k in (1,2,3)};p['by_pair']={str(k):pooled(v,lambda d,a:obs[d,a]['harmonic_pair']==k) for k in range(1,7)};results[v]=p
  for name,q in results.items():
   if not name.startswith('MASK'):continue
   checks=dict(accuracy=100*(q['accuracy']-base_score['accuracy'])>=-.5-1e-12,F1=100*(q['macro_F1']-base_score['macro_F1'])>=-.5-1e-12,retention=q['raw_correct_retention']>=.99)
   for group in ('RX','TX','day'):checks[group]=all(100*(r['accuracy']-base_score['by_'+group][k]['accuracy'])>=-1-1e-12 for k,r in q['by_'+group].items())
   identity_checks.append(dict(view=name,checks=checks,qualified=all(checks.values())))
  iq=[r['relative_IQ_RMS'] for r in obs.values()];err=sum(r['boundary_error_energy'] for r in obs.values());change=sum(r['change_energy'] for r in obs.values());dsp=dict(median=statistics.median(iq),maximum=max(iq),boundary_error_to_change=err/change,power_match_error_max=max(r['power_match_relative_error'] for r in obs.values()),cap_fraction=sum(r['cap']<1 for r in obs.values())/len(obs),IQ_change=summary(iq),matched_gain=summary([r['matched_gain'] for r in obs.values()]),reference_window_relative_RMS=summary([r['reference_window_relative_RMS'] for r in obs.values()]))
  independent=dict(identity=all(r['qualified'] for r in identity_checks),nontrivial=dsp['median']>=.01,bounded=dsp['maximum']<=.100001,boundary=dsp['boundary_error_to_change']<=.01,power_match=dsp['power_match_error_max']<=1e-6)
  for v,key in (('raw','raw0'),('MASK','MASK'),('GAIN','GAIN')):
   for k in ('accuracy','macro_F1'):assert abs(expected[v][k]-results[key][k])<1e-12
  assert expected['qualified']==all(independent.values())
  models[str(seed)]=dict(model_seed=seed,raw=base_score,MASK=results['MASK'],GAIN=results['GAIN'],draws={k:v for k,v in results.items() if k not in ('raw0','MASK','GAIN')},identity_checks=identity_checks,DSP=dsp,qualified=all(independent.values()),checks=independent,NN=NN,source_physical_L=5040,source_RX=[3,4,6,8],source_cells=72,wall_seconds_original=expected['wall_seconds'],independent_result_match=True,IQ_read=0,CP_reads=0,updates=0,target_access=False)
  same.append([(d,a,r['harmonic_pair'],r['relative_IQ_RMS'],r['matched_gain']) for (d,a),r in sorted(obs.items())])
 assert same[0]==same[1]
 result=dict(status='OFFLINE_FULL_RECOUNT_COMPLETE',models=models,qualified=all(v['qualified'] for v in models.values()),source_NN_original=90720,independent_NN=0,independent_IQ=0,independent_CP=0,updates=0,identical_two_model_operator_streams=True,target_access=False,scope='Frozen fit-source identity only;not heldRX/Ag/As/urban/joint/crossseed training efficacy',wall_seconds_offline=time.perf_counter()-start)
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();r=run(a.root,a.out);print(json.dumps(dict(status=r['status'],qualified=r['qualified'],source_NN_original=r['source_NN_original'],offline_NN=0,wall_seconds=r['wall_seconds_offline'])))
