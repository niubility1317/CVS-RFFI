"""Independent L_s truth-last re-score using fixed Phase1 source prediction arrays."""
import pathlib,sys,json,hashlib,numpy as np
R=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');score=R/'runs'/sys.argv[1]/'source_ls_truth_score.json';s=json.loads(score.read_text());assert s['status']=='SOURCE_LS_TRUTH_LAST_COMPLETE' and s['fixed_predictions_before_truth'] and s['truth_role']=='L_s only' and s['U_s_truth_reads']==s['V_reads']==s['test_reads']==0
root=R/'runs'/s['prediction_run'];path=root/'source_predictions.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==s['prediction_sha256'];data=np.load(path,allow_pickle=False);ids=data['indices'];split=data['split'];rx=data['rx'];scene=data['scene'];v=data['values'];assert v.shape==(11761,15) and len(ids)==11761
cache=R/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache';physical=np.load(cache/'train/physical_ids.npy',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False);meta=R/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/L_s_metadata.npz';contract=json.loads((meta.parent/'source_contract.json').read_text())
with np.load(meta,allow_pickle=False) as z:physical_ls=z['ids'];truth_ls=z['y']
assert physical_ls.tolist()==contract['role_ids']['L_s'] and len(physical_ls)==6300 and np.all((truth_ls>=0)&(truth_ls<6));truth_map={str(k):int(v) for k,v in zip(physical_ls,truth_ls)};known=roles[ids]=='L_s';assert int(known.sum())==s['L_s_count'] and all(str(k) in truth_map for k in physical[ids[known]]) and all(str(k) not in truth_map for k in physical[ids[~known]]);truth=np.full(len(ids),-1,dtype=np.int64);truth[known]=[truth_map[str(k)] for k in physical[ids[known]]]
active=v[:,3]>.8;pred={'clean':v[:,0],'raw':v[:,5],'dura':v[:,2],'parent':np.where(active,v[:,7],v[:,5]),'candidate':np.where(active,v[:,13],v[:,5])};checks=0
for group,pnum,receivers in [('RX4_6',1,(4,6)),('RX8',2,(8,))]:
 for view,cell in s['results'][group].items():
  mask=known&(split==pnum)&np.isin(rx,receivers)&((scene!='clean') if view=='all_impaired' else (scene==view));assert int(mask.sum())==cell['count'] and int((mask&active).sum())==cell['active'];correct={k:(q==truth) for k,q in pred.items()};assert int((mask&~correct['clean']).sum())==cell['clean_f_pseudo_wrong']
  for name,c in correct.items():
   t=cell[name];assert int((mask&c).sum())==t['correct'] and abs(float(c[mask].mean()*100)-t['accuracy_percent'])<1e-10
   for ref in ('parent','dura'):
    assert int((mask&c&~correct[ref]).sum())==t['repairs_vs_'+ref] and int((mask&~c&correct[ref]).sum())==t['damages_vs_'+ref]
  checks+=1
assert checks==16;print(json.dumps({'status':'PASS','candidate':s['candidate'],'labeled_source_records':int(known.sum()),'cells':checks,'prediction_sha256':s['prediction_sha256'],'method':'independent L_s physical-ID join and paired count recomputation','U_s_truth_reads':0,'V_reads':0,'test_reads':0}))
