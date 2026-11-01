import pathlib,sys,numpy as np,torch,json
from torch.nn import functional as F
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet')
for n in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release','20261102-phase1-phase-shift-student-s2026092701-r01-release/experiments/phase1_phase_shift_restorer']:
 sys.path.insert(0,str(root/'releases'/n))
import phase_shift_source as s
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
torch.set_num_threads(2);cache,clean,rx,train,_,_=s.source_contract(root);received,scene=s.epoch_data(cache,12);rng=np.random.default_rng(2026111755)
views=['practical_high','practical_mid_urban','practical_low_urban'];idx=np.concatenate([rng.choice(train[scene[train]==v],128,replace=False) for v in views]);teacher,_=core.teacher(root,'cpu');dura=prior.frozen_dura(root,'cpu');m,gate=s.initial_models(root,'cpu');m.eval();shifts=(-24,-12,0,12,24);rots=(1+0j,0+1j,-1+0j,0-1j);rows=[]
with torch.no_grad():
 for start in range(0,len(idx),16):
  ix=idx[start:start+16];x=core.tensor(received[ix],'cpu');cl=core.tensor(clean[ix],'cpu');p=F.pad(x,(24,24),mode='reflect');cleanprob=teacher.classify_features(teacher.features(cl)).softmax(1);target=cleanprob.argmax(1);valid=cleanprob.max(1).values>=.95;d=teacher.classify_features(teacher.features(dura(x))).argmax(1);a=teacher.classify_features(teacher.features(m.anchor(x))).argmax(1);allp=[]
  for shift in shifts:
   q=p[:,:,24+shift:24+shift+256];z=torch.complex(q[:,0],q[:,1])
   for rot in rots:
    zr=z*rot;w=torch.stack((zr.real,zr.imag),1);allp.append(teacher.classify_features(teacher.features(w)).softmax(1))
  prob=torch.stack(allp,1);pred=prob.argmax(2);confidence=prob.max(2).values;best=confidence.argmax(1);raw=pred[:,8];ensemble=prob.mean(1);consensus=ensemble.argmax(1);chosen=prob.gather(2,consensus[:,None,None].expand(-1,len(shifts)*len(rots),1)).squeeze(2).argmax(1);row=torch.arange(len(ix));selected=pred[row,chosen];most=pred[row,best];oracle=(pred==target[:,None]).any(1);rows.append(torch.stack((valid.float(),(raw==target).float(),(d==target).float(),(a==target).float(),(most==target).float(),(consensus==target).float(),(selected==target).float(),oracle.float()),1))
v=np.from_dlpack(torch.cat(rows).contiguous());out={}
for view in views:
 g=v[scene[idx]==view];g=g[g[:,0]>.5];out[view]={'valid_count':len(g),'raw':int(g[:,1].sum()),'dura':int(g[:,2].sum()),'anchor':int(g[:,3].sum()),'max_conf_candidate':int(g[:,4].sum()),'mean_probability_class':int(g[:,5].sum()),'consensus_selected_candidate':int(g[:,6].sum()),'pseudo_label_oracle':int(g[:,7].sum())}
print(json.dumps({'source_train_only':True,'per_scene_records':128,'views':views,'transforms':'five reflect time shifts and four global phase rotations; candidates chosen by model outputs only','pseudo_target':'clean ground F confidence >=0.95; no source labels','nondeployable_oracle':'uses pseudo target per sample; bounds only these 20 outputs','no_TX_labels':True,'no_U_truth':True,'no_V':True,'no_test':True,'results':out},allow_nan=False))
