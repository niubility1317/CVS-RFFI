"""Audit source class counts via sorted physical-ID joins and set algebra."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--commit',required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError('Immutable source diagnostic audit required')
 result=json.loads((a.source/'source_class_counts.json').read_text());runs=a.root/'runs';names=['20261102-phase1-unroll-source-replay-s2026092701-r01','20261102-phase1-supervised-unroll-source-replay-s2026092701-r01','20261102-phase1-spectral-synthesis-full-source-s2026092701-r02'];paths=[runs/n/'source_predictions.npz' for n in names];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();hashes=[sha(p) for p in paths]
 if result.get('status')!='SOURCE_CLASS_DIAGNOSTIC_COMPLETE' or result.get('commit')!=a.commit or result.get('parent_sha256')!=hashes or result.get('L_s_scored')!=1047 or any(result.get(k)!=0 for k in ('U_s_truth_reads','V_reads','test_reads')):raise ValueError('Fixed source diagnostic provenance required')
 with np.load(paths[0],allow_pickle=False) as p,np.load(paths[1],allow_pickle=False) as s,np.load(paths[2],allow_pickle=False) as z:
  ids=p['indices'];rx=p['rx'];scene=p['scene'];v=p['values'];sv=s['candidate_values'];fv=z['spectral_values']
  if any(not np.array_equal(p[k],q[k]) for q in (s,z) for k in ('indices','rx','scene')) or not np.array_equal(v,s['prior_values']) or not np.array_equal(v,z['prior_values']):raise ValueError('Fixed source arrays differ')
 cache=runs/'20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache';physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False);meta=runs/'20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/L_s_metadata.npz';contract=json.loads((meta.parent/'source_contract.json').read_text())
 with np.load(meta,allow_pickle=False) as z:ls_ids=z['ids'];ls_truth=z['y']
 if ls_ids.tolist()!=contract['role_ids']['L_s'] or len(ls_ids)!=6300 or not np.all((ls_truth>=0)&(ls_truth<6)):raise ValueError('Exact L_s role contract required')
 positions=np.flatnonzero(roles[ids]=='L_s');order=np.argsort(ls_ids);sorted_ids=ls_ids[order];positions_in_ls=np.searchsorted(sorted_ids,physical[ids[positions]])
 if len(positions)!=1047 or not np.array_equal(sorted_ids[positions_in_ls],physical[ids[positions]]):raise ValueError('L_s physical-ID join failed')
 truth={int(i):int(ls_truth[order[j]]) for i,j in zip(positions,positions_in_ls)}
 def pred(name,i):return int(v[i,5] if name=='raw' else v[i,7] if name=='physical' else v[i,13] if name=='unroll' else sv[i,0] if name=='single' else sv[i,2] if name=='multiview' else fv[i,0])
 checked=0;fields=0
 for group,receivers in [('RX4_6',{4,6}),('RX8',{8})]:
  for view in ['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','all_impaired']:
   for label in range(6):
    members=[i for i,y in truth.items() if int(rx[i]) in receivers and y==label and (scene[i]!='clean' if view=='all_impaired' else scene[i]==view)];cell=result['cells'][f'{group}:{view}:{label}'];assert cell['count']==len(members);fields+=1;raw={i for i in members if pred('raw',i)==label};physical_correct={i for i in members if pred('physical',i)==label}
    for name in ('raw','physical','unroll','single','multiview','spectral'):
     good={i for i in members if pred(name,i)==label};item=cell['methods'][name];assert item['correct']==len(good) and item['repairs_vs_raw']==len(good-raw) and item['damages_vs_raw']==len(raw-good) and item['repairs_vs_physical']==len(good-physical_correct) and item['damages_vs_physical']==len(physical_correct-good);fields+=5
    checked+=1
 if checked!=96:raise ValueError('96 group/view/class cells required')
 out=dict(status='PASS',source_only=True,L_s_scored=len(truth),cells=checked,fields_recomputed=fields,parent_sha256=hashes,method='independent sorted physical-ID label join and paired set counts',U_s_truth_reads=0,V_reads=0,test_reads=0)
 with a.output.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
