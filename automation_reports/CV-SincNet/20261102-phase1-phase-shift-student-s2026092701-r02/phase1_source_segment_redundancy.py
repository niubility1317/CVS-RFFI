import pathlib,sys,numpy as np,torch,json
from torch.nn import functional as F
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet')
for n in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release','20261102-phase1-phase-shift-student-s2026092701-r02-release/experiments/phase1_phase_shift_restorer']:sys.path.insert(0,str(root/'releases'/n))
import phase_shift_source as s
from experiments.pamf_phase1_flow import experiment as core
torch.set_num_threads(2);cache,clean,rx,train,_,_=s.source_contract(root);received,scene=s.epoch_data(cache,14);rng=np.random.default_rng(2026112955);views=[v for v in s.VIEWS if v!='clean'];ids=np.concatenate([rng.choice(train[scene[train]==v],128,replace=False) for v in views]);pairs=[(0,80),(80,160),(0,160)];rows=[]
def cor(z,a,b):
 x=z[:,a:a+64];y=z[:,b:b+64];return (x.conj()*y).sum(1).abs()/(x.abs().square().sum(1)*y.abs().square().sum(1)).clamp_min(1e-20).sqrt()
with torch.no_grad():
 for st in range(0,len(ids),64):
  i=ids[st:st+64];x=core.tensor(received[i],'cpu');y=core.tensor(clean[i],'cpu');x=F.pad(x,(64,64),mode='reflect')[:,:,88:344];zx=torch.complex(x[:,0],x[:,1]);zy=torch.complex(y[:,0],y[:,1]);rows.append(torch.stack([*(cor(zy,a,b) for a,b in pairs),*(cor(zx,a,b) for a,b in pairs)],1))
v=np.from_dlpack(torch.cat(rows).contiguous());out={}
for name in views:
 a=v[scene[ids]==name];out[name]={'count':len(a),'pairs':{str(pair):{'clean_q10_median_q90':np.percentile(a[:,i],[10,50,90]).tolist(),'received_q10_median_q90':np.percentile(a[:,3+i],[10,50,90]).tolist()} for i,pair in enumerate(pairs)}}
print(json.dumps({'source_train_only':True,'records':len(ids),'segment_length':64,'TX_labels_read':0,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'results':out},allow_nan=False))
