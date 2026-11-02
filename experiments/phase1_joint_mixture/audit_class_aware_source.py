"""Independently replay fixed receiver-fold selections and source score."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from class_aware_router import ClassAwareRouter

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--score',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 m=json.loads((a.source/'result.json').read_text());score=json.loads(a.score.read_text());path=a.source/'source_predictions.npz'
 if score['status']!='SOURCE_ROUTER_SCORE_COMPLETE' or sha(path)!=m['source_output_sha256'] or sha(path)!=score['prediction_sha256']:raise ValueError('Fixed output binding failed')
 with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 parents=[]
 for kind,run in [('complementary','20261102-phase1-complementary-full-source-s2026092701-r01'),('free','20261102-phase1-free-mixture-full-source-s2026092701-r01')]:
  f=a.root/'runs'/run/'source_predictions.npz';audit=json.loads((a.root/'runs/20261102-phase1-router-full-source-audit-s2026092701-r01'/(kind+'.json')).read_text())
  if sha(f)!=m['source_prediction_sha256'][kind] or audit['prediction_sha256']!=sha(f) or audit['status']!='PASS':raise ValueError('Candidate parent audit failed')
  with np.load(f,allow_pickle=False) as z:parents.append({k:z[k].copy() for k in z.files})
 c,f=parents
 for k in ('indices','rx','scene','prior_values'):
  if not np.array_equal(d[k],c[k]) or not np.array_equal(d[k],f[k]):raise ValueError('Candidate alignment failed '+k)
 expected=np.column_stack((c['prior_values'][:,5].astype(np.int64),c['candidate_predictions'],f['candidate_prediction']))
 if not np.array_equal(expected,d['candidate_predictions']):raise ValueError('Candidate reconstruction failed')
 torch.set_num_threads(2);checked=0
 for name,train,ev in [('train4_eval6',[4],[6]),('train6_eval4',[6],[4]),('train46_eval8',[4,6],[8])]:
  ckpath=a.source/(name+'.pt')
  if sha(ckpath)!=m['fold_model_sha256'][name]:raise ValueError('Model hash failed')
  ck=torch.load(ckpath,map_location='cpu',weights_only=False);fold=ck['fold']
  if ck['commit']!=m['commit'] or ck['candidate_names']!=m['candidate_names'] or ck['source_prediction_sha256']!=m['source_prediction_sha256'] or fold['train_receivers']!=train or fold['eval_receivers']!=ev:raise ValueError('Fold binding failed')
  model=ClassAwareRouter();model.load_state_dict(ck['model'],strict=True);model.eval();sel=np.isin(d['rx'],ev)
  with torch.no_grad():ch,y=model.choose(torch.tensor(expected[sel].tolist(),dtype=torch.long))
  if not np.array_equal(np.from_dlpack(ch.contiguous()),d['choice'][sel]) or not np.array_equal(np.from_dlpack(y.contiguous()),d['prediction'][sel]):raise ValueError('Fold prediction replay failed')
  checked+=int(sel.sum())
 if checked!=11761:raise ValueError('Receiver coverage failed')
 cache=a.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train';ids=d['indices'];physical=np.load(cache/'physical_ids.npy',allow_pickle=False);roles=np.load(cache/'roles.npy',allow_pickle=False)
 if not np.array_equal(physical[ids],d['physical_ids']):raise ValueError('Physical IDs failed')
 meta=a.root/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/L_s_metadata.npz'
 with np.load(meta,allow_pickle=False) as z:keys=z['ids'];labels=z['y']
 lookup=dict(zip(map(str,keys),map(int,labels)));names=m['candidate_names']+['router'];y=np.array([lookup.get(str(k),-1) for k in d['physical_ids']]);known=roles[ids]=='L_s'
 if int(known.sum())!=1047 or not np.array_equal(y>=0,known):raise ValueError('Authorized labels do not bind roles')
 scenes=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired'];checks=0
 for g,receivers in [('RX4_6',(4,6)),('RX8',(8,))]:
  for view in scenes:
   ix=[i for i in range(len(ids)) if int(d['rx'][i]) in receivers and (d['scene'][i]!='clean' if view=='all_impaired' else d['scene'][i]==view)];cell=score['results'][g][view];pseudo=[i for i in ix if d['prior_values'][i,1]>=.95];ls=[i for i in ix if known[i]]
   assert cell['count']==len(ix) and cell['pseudo_count']==len(pseudo) and cell['L_s_count']==len(ls);checks+=3
   for j,name in enumerate(m['candidate_names']):assert cell['selected_counts'][name]==sum(int(d['choice'][i])==j for i in ix);checks+=1
   for label,rows,target in [('pseudo',pseudo,d['prior_values'][:,0].astype(int)),('L_s',ls,y)]:
    assert cell[label+'_oracle_correct']==sum(int(target[i] in expected[i]) for i in rows);checks+=1
    for j,name in enumerate(names):
     pred=expected[:,j] if j<5 else d['prediction'];base=expected[:,4];correct=sum(pred[i]==target[i] for i in rows);rep=sum(pred[i]==target[i] and base[i]!=target[i] for i in rows);harm=sum(pred[i]!=target[i] and base[i]==target[i] for i in rows);item=cell[name]
     assert (item[label+'_correct'],item[label+'_repairs_vs_free'],item[label+'_damages_vs_free'])==(correct,rep,harm);checks+=3
 out={'status':'PASS','prediction_sha256':sha(path),'model_replayed':3,'records_replayed':checked,'independent_metrics_checked':checks,'L_s_records':int(known.sum()),'U_s_truth_reads':0,'V_reads':0,'test_reads':0}
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x',encoding='utf8') as h:json.dump(out,h,indent=2);h.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
