import pathlib,sys,torch,numpy as np
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');sys.path.insert(0,str(root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release'));sys.path.insert(0,str(root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'));sys.path.insert(0,str(root/'releases/20261102-phase1-teacher-student-s2026092701-r01-release/experiments/phase1_teacher_student'))
from phase1_teacher_student_source import source_contract,epoch_data
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
from torch.nn import functional as F
torch.set_num_threads(2);cache,clean,rx,train,_,_=source_contract(root);received,scene=epoch_data(cache,4);pool=train[scene[train]!='clean'];ids=np.random.default_rng(2026111458).choice(pool,size=384,replace=False);dura=prior.frozen_dura(root,'cpu');rows=[]
with torch.no_grad():
 for start in range(0,len(ids),32):
  idx=ids[start:start+32];x=core.tensor(received[idx],'cpu');d=dura(x);p=F.pad(x,(64,64),mode='reflect');z=torch.complex(p[:,0],p[:,1]).unfold(-1,256,1)[:,:,64:-64];ref=torch.complex(d[:,0,64:-64],d[:,1,64:-64])[:,None,:];g=(z.conj()*ref).sum(-1)/z.abs().square().sum(-1).clamp_min(1e-20);e=(g[:,:,None]*z-ref).abs().square().mean(-1);best=e.argmin(1);gain=g.gather(1,best[:,None]).squeeze(1);raw=torch.complex(x[:,0,64:-64],x[:,1,64:-64]);v=torch.stack(((best-64).float(),gain.abs(),e.gather(1,best[:,None]).squeeze(1),ref[:,0].abs().square().mean(1),(raw-ref[:,0]).abs().square().mean(1)),1);rows.append(v)
v=np.from_dlpack(torch.cat(rows).contiguous());print('SOURCE TRAIN 384 no TX/U_s truth/V/test');
for view in np.unique(scene[ids]):
 a=v[scene[ids]==view];print(view,'n',len(a),'delay median',float(np.median(a[:,0])),'delay 10/90',np.percentile(a[:,0],[10,90]).tolist(),'delay mode',int(np.bincount(a[:,0].astype(int)+64,minlength=129).argmax()-64),'zero',int((a[:,0]==0).sum()),'abs_delay<=2',int((abs(a[:,0])<=2).sum()),'gain median',float(np.median(a[:,1])),'aligned NMSE',float(np.mean(a[:,2]/np.maximum(a[:,3],1e-12))),'raw NMSE',float(np.mean(a[:,4]/np.maximum(a[:,3],1e-12))))
