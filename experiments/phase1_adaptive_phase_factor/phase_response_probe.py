import pathlib,sys,numpy as np,torch
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');sys.path.insert(0,str(root/'releases/20261102-phase1-teacher-student-s2026092701-r01-release/experiments/phase1_teacher_student'))
from phase1_teacher_student_source import source_contract,epoch_data
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
torch.set_num_threads(2);cache,clean,rx,training,valid,hold=source_contract(root);received,scene=epoch_data(cache,2);rng=np.random.default_rng(2026111352);pool=training[scene[training]!='clean'];ids=rng.choice(pool,size=256,replace=False);dura=prior.frozen_dura(root,'cpu')
with torch.no_grad():
 for theta in (1.5707963267948966,3.141592653589793):
  aa=[];bb=[]
  for start in range(0,len(ids),32):
   x=core.tensor(received[ids[start:start+32]],'cpu');z=torch.complex(x[:,0],x[:,1]);rot=torch.polar(torch.ones_like(z.real),torch.full_like(z.real,theta));v=z*rot;d=dura(x);dr=dura(torch.stack((v.real,v.imag),1));dz=torch.complex(d[:,0],d[:,1]);drz=torch.complex(dr[:,0],dr[:,1]);den=dz.abs().square().mean(1).clamp_min(1e-12);aa.append((drz-dz).abs().square().mean(1)/den);bb.append((drz-dz*rot).abs().square().mean(1)/den)
  a=np.from_dlpack(torch.cat(aa).contiguous());b=np.from_dlpack(torch.cat(bb).contiguous())
  for v in np.unique(scene[ids]):
   mask=scene[ids]==v;print('ANGLE',round(theta,3),'SCENE',v,'N',int(mask.sum()),'invariant_median',round(float(np.median(a[mask])),4),'equivariant_median',round(float(np.median(b[mask])),4))
print('SOURCE_TRAIN_ONLY=256 TX_labels_read=0 U_s_truth_reads=0 V_reads=0 test_reads=0')
