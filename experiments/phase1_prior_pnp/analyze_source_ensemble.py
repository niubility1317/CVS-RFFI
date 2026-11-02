"""Post-hoc source pseudo-label oracle; no target or TX truth."""
import pathlib,numpy as np,json,hashlib
R=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');root=R/'runs/20261102-phase1-independent-prior-pnp-step-diagnostic-s2026092701-r02';meta=json.loads((root/'prediction_complete.json').read_text());path=root/'source_predictions.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256'] and meta['count']==2048 and meta['TX_truth_reads']==meta['U_s_truth_reads']==meta['V_reads']==meta['test_reads']==0
with np.load(path,allow_pickle=False) as z:v=z['values'];part=z['part'];scene=z['scene'];rx=z['rx']
assert v.shape==(2048,23);methods={'raw':3,'parent':9,'observation':12,'prior':15,'alternating':18};out={}
for group,sel in [('RX4_6',part==1),('RX8',part==2)]:
 for sc in ('all','practical_low_urban'):
  mask=sel&(v[:,1]>=.95)
  if sc!='all':mask&=scene==sc
  d=v[mask];correct={k:d[:,i]==d[:,0] for k,i in methods.items()};anyok=np.logical_or.reduce(list(correct.values()));allok=np.logical_and.reduce(list(correct.values()));par=correct['parent'];unionadd=int(np.count_nonzero(~par&anyok));unique={k:int(np.count_nonzero(correct[k]&~np.logical_or.reduce([b for kk,b in correct.items() if kk!=k]))) for k in methods};none=int(np.count_nonzero(~anyok));patterns={}
  for name in methods:patterns[name]={'pseudo_correct':int(correct[name].sum()),'repairs_vs_parent':int(np.count_nonzero(~par&correct[name])),'damages_vs_parent':int(np.count_nonzero(par&~correct[name])),'uniquely_correct':unique[name]}
  out[group+':'+sc]={'trusted_count':len(d),'union_oracle_correct':int(anyok.sum()),'union_extra_vs_parent':unionadd,'all_correct':int(allok.sum()),'none_correct':none,'methods':patterns}
print(json.dumps({'status':'SOURCE_ORACLE_ANALYSIS','origin_sha256':meta['sha256'],'source_only':True,'target':'high-confidence clean-F pseudo-label, not true identity','union_rule':'label oracle across five already fixed independent candidate predictions, not deployable and not a trained router','DURA_in_union':False,'TX_truth_reads':0,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'groups':out},allow_nan=False))
