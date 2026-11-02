"""Independent fixed-model and source-label audit for gate-aware router."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from gate_aware_router import GateAwareRouter

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--score',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 m=json.loads((a.source/'result.json').read_text());score=json.loads(a.score.read_text());path=a.source/'source_predictions.npz';parent=a.root/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01/source_predictions.npz'
 if m.get('status')!='SOURCE_GATE_ROUTER_TRAINING_COMPLETE' or m.get('commit')!='9dac8eaa0a35e352652791bf0080ade59751af09' or sha(path)!=m.get('source_output_sha256') or sha(path)!=score.get('prediction_sha256') or score.get('status')!='SOURCE_GATE_ROUTER_SCORE_COMPLETE' or sha(parent)!=m.get('prior_sha256'):raise ValueError('Exact fixed model and source output required')
 with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 with np.load(parent,allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
 for k in ('indices','rx','scene'):
  if not np.array_equal(d[k],p[k]):raise ValueError('Source parent identity mismatch '+k)
 if not np.array_equal(d['prior_values'],p['values']):raise ValueError('Source parent values differ')
 prior=p['values'];gate=prior[:,3];on=gate>.8;raw=prior[:,5].astype(np.int64);expected=np.column_stack((raw,np.where(on,prior[:,7],raw),*[np.where(on,prior[:,i],raw) for i in (9,11,13)])).astype(np.int64)
 if not np.array_equal(expected,d['candidate_predictions']) or expected.shape!=(11761,5):raise ValueError('Candidate reconstruction failed')
 torch.set_num_threads(1);replayed=0
 for name,train,ev in [('train4_eval6',[4],[6]),('train6_eval4',[6],[4]),('train46_eval8',[4,6],[8])]:
  ckpath=a.source/(name+'.pt')
  if sha(ckpath)!=m['fold_model_sha256'][name]:raise ValueError('Fold model hash mismatch')
  ck=torch.load(ckpath,map_location='cpu',weights_only=False);fold=ck['fold']
  if ck['commit']!=m['commit'] or ck['candidate_names']!=m['candidate_names'] or ck['prior_sha256']!=m['prior_sha256'] or fold['train_receivers']!=train or fold['eval_receivers']!=ev:raise ValueError('Fold source binding mismatch')
  net=GateAwareRouter();net.load_state_dict(ck['model'],strict=True);net.eval();sel=np.isin(d['rx'],ev)
  with torch.no_grad():choice,pred=net.choose(torch.tensor(expected[sel].tolist(),dtype=torch.long),torch.tensor(gate[sel].tolist(),dtype=torch.float32))
  if not np.array_equal(np.from_dlpack(choice.contiguous()),d['choice'][sel]) or not np.array_equal(np.from_dlpack(pred.contiguous()),d['prediction'][sel]):raise ValueError('Independent source model replay mismatch')
  replayed+=int(sel.sum())
 if replayed!=11761:raise ValueError('Receiver fold coverage failed')
 runs=a.root/'runs';cache=runs/'20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train';ids=d['indices'];physical=np.load(cache/'physical_ids.npy',allow_pickle=False);roles=np.load(cache/'roles.npy',allow_pickle=False);meta=runs/'20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/L_s_metadata.npz';contract=json.loads((meta.parent/'source_contract.json').read_text())
 if not np.array_equal(physical[ids],d['physical_ids']):raise ValueError('Source physical IDs differ')
 with np.load(meta,allow_pickle=False) as z:keys=z['ids'].copy();labels=z['y'].copy()
 if keys.tolist()!=contract['role_ids']['L_s'] or len(keys)!=6300:raise ValueError('Exact source L_s contract required')
 key=np.asarray([str(x) for x in keys]);order=np.argsort(key);ks=key[order];query=np.asarray([str(x) for x in physical[ids]]);pos=np.searchsorted(ks,query);present=(pos<len(ks))&(ks[np.minimum(pos,len(ks)-1)]==query);known=roles[ids]=='L_s'
 if len(set(key.tolist()))!=6300 or int(known.sum())!=1047 or not np.array_equal(present,known) or not np.all(known|(roles[ids]=='U_s')):raise ValueError('Independent source role join failed')
 truth=np.full(len(ids),-1,dtype=np.int64);truth[known]=labels[order[pos[known]]];names=m['candidate_names']+['router'];fields=0
 for group,receivers in [('RX4_6',(4,6)),('RX8',(8,))]:
  for view in ['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired']:
   ix=[i for i in range(len(ids)) if int(d['rx'][i]) in receivers and (d['scene'][i]!='clean' if view=='all_impaired' else d['scene'][i]==view)];pseudo=[i for i in ix if prior[i,1]>=.95];ls=[i for i in ix if known[i]];cell=score['results'][group][view]
   if (cell['count'],cell['pseudo_count'],cell['L_s_count'])!=(len(ix),len(pseudo),len(ls)):raise ValueError('Cell counts differ')
   fields+=3
   for j,name in enumerate(m['candidate_names']):
    if cell['selected_counts'][name]!=sum(int(d['choice'][i])==j for i in ix):raise ValueError('Selection counts differ')
    fields+=1
   for label,rows,target in [('pseudo',pseudo,prior[:,0].astype(int)),('L_s',ls,truth)]:
    if cell[label+'_oracle_correct']!=sum(int(target[i] in expected[i]) for i in rows):raise ValueError('Label-aided oracle mismatch')
    fields+=1
    for j,name in enumerate(names):
     pred=expected[:,j] if j<5 else d['prediction'];base=expected[:,1];good={i for i in rows if pred[i]==target[i]};base_good={i for i in rows if base[i]==target[i]};item=cell[name]
     if (item[label+'_correct'],item[label+'_repairs_vs_physical'],item[label+'_damages_vs_physical'])!=(len(good),len(good-base_good),len(base_good-good)):raise ValueError('Independent source score mismatch')
     fields+=3
 out=dict(status='PASS',records_replayed=replayed,model_replayed=3,L_s_scored=int(known.sum()),independent_fields=fields,prediction_sha256=sha(path),prior_sha256=sha(parent),U_s_truth_reads=0,V_reads=0,test_reads=0,method='fold model replay, sorted physical-ID join and independent set counts')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
