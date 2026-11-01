import pathlib,sys,numpy as np,torch
from torch.nn import functional as F
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');sys.path.insert(0,str(root/'releases/20261102-phase1-teacher-student-s2026092701-r01-release/experiments/phase1_teacher_student'))
from phase1_teacher_student_source import source_contract,epoch_data,align
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
torch.set_num_threads(2);cache,clean,rx,training,valid,hold=source_contract(root);received,scene=epoch_data(cache,1);rng=np.random.default_rng(2026111351);pool=training[scene[training]!='clean'];ids=rng.choice(pool,size=512,replace=False);dura=prior.frozen_dura(root,'cpu');f,_=core.teacher(root,'cpu');results=[]
with torch.no_grad():
 for start in range(0,len(ids),32):
  ix=ids[start:start+32];x=core.tensor(received[ix],'cpu');y=core.tensor(clean[ix],'cpu');d=dura(x);da,_=align(d,x);norm=x.square().mean((1,2)).clamp_min(1e-12)
  raw_nmse=(d-x).square().mean((1,2))/norm;aligned_nmse=(da-x).square().mean((1,2))/norm
  py=F.softmax(f.classify_features(f.features(y)),1);truth=py.argmax(1);valid=py.max(1).values>=.95
  pd=f.classify_features(f.features(d)).argmax(1);pa=f.classify_features(f.features(da)).argmax(1)
  _,e=align(d,y);_,ea=align(da,y);den=y[:,:,64:-64].square().sum(1).mean(1).clamp_min(1e-12)
  results.append(torch.stack((raw_nmse,aligned_nmse,(e/den),(ea/den),valid.float(),(pd==truth).float(),(pa==truth).float()),1))
a=torch.cat(results);mask=a[:,4].bool();print('SOURCE_TRAIN_ONLY',len(a),'valid_pseudo',int(mask.sum()),'raw_teacher_NMSE',float(a[:,0].mean()),'aligned_teacher_NMSE',float(a[:,1].mean()),'teacher_clean_IQ',float(a[:,2].mean()),'aligned_teacher_clean_IQ',float(a[:,3].mean()),'DURA_F_agreement',int(a[mask,5].sum()),'aligned_F_agreement',int(a[mask,6].sum()),'TX_labels_read=0 U_s_truth_reads=0 V_reads=0 test_reads=0')
