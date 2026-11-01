import pathlib,sys,torch,numpy as np
from torch.nn import functional as F
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');sys.path.insert(0,str(root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release'));sys.path.insert(0,str(root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'));sys.path.insert(0,str(root/'releases/20261102-phase1-teacher-student-s2026092701-r01-release/experiments/phase1_teacher_student'));sys.path.insert(0,str(root/'releases/20261102-phase1-adaptive-phase-factor-s2026092701-r01-release/experiments/phase1_adaptive_phase_factor'))
from phase1_teacher_student_source import source_contract,epoch_data
from phase_factor_model import AdaptivePhaseFactor
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
torch.set_num_threads(2);cache,clean,rx,train,_,_=source_contract(root);received,scene=epoch_data(cache,4);pool=train[scene[train]!='clean'];ids=np.random.default_rng(2026111458).choice(pool,size=384,replace=False);dura=prior.frozen_dura(root,'cpu');f,_=core.teacher(root,'cpu');m=AdaptivePhaseFactor();ck=torch.load(root/'runs/20261102-phase1-adaptive-phase-factor-s2026092701-r01/source/selected.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);m.eval().requires_grad_(False);rows=[]
with torch.no_grad():
 for start in range(0,len(ids),32):
  ix=ids[start:start+32];x=core.tensor(received[ix],'cpu');cl=core.tensor(clean[ix],'cpu');d=dura(x);s=F.pad(x,(64,64),mode='reflect')[:,:,88:344];phase=m(x);z=torch.complex(s[:,0],s[:,1])*phase[:,None];sp=torch.stack((z.real,z.imag),1);waves=(x,s,sp,d);logits=[f.classify_features(f.features(y)) for y in (cl,*waves)];probs=logits[0].softmax(1);target=probs.argmax(1);good=probs.max(1).values>=.95;power=d[:,:,64:-64].square().mean((1,2)).clamp_min(1e-12);errs=[(w[:,:,64:-64]-d[:,:,64:-64]).square().mean((1,2))/power for w in waves[:3]];rows.append(torch.stack((good.float(),*((v.argmax(1)==target).float() for v in logits[1:]),*errs),1))
v=np.from_dlpack(torch.cat(rows).contiguous());print('SOURCE TRAIN 384 SAME SAMPLE fixed shift estimated on same IDs; no TX/U_s truth/V/test')
for name in np.unique(scene[ids]):
 a=v[scene[ids]==name];b=a[a[:,0]>.5];print(name,'n',len(a),'pseudo_valid',len(b),'raw_shift_shiftphase_dura_correct',b[:,1:5].sum(0).astype(int).tolist(),'central_NMSE',a[:,5:8].mean(0).tolist())
