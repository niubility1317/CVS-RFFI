import sys,pathlib,torch,numpy as np,json
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet')
for n in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release','20261102-phase1-phase-shift-student-s2026092701-r02-release/experiments/phase1_phase_shift_restorer']:sys.path.insert(0,str(root/'releases'/n))
import phase_shift_source as s
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
torch.set_num_threads(2);torch.backends.mkldnn.enabled=False;torch.manual_seed(2026111954);cache,clean,rx,train,prior_val,prior_hold=s.source_contract(root);val,held=s.new_splits(cache,rx,train,prior_val,prior_hold);rng=np.random.default_rng(2026111955);chosen=rng.choice(train,size=8192,replace=False);received,scene=s.epoch_data(cache,7);f,_=core.teacher(root,'cpu');features=[];targets=[];conf=[]
with torch.no_grad():
 for start in range(0,len(chosen),64):
  i=chosen[start:start+64];x=core.tensor(received[i],'cpu');y=core.tensor(clean[i],'cpu');features.append(f.features(x).cpu());p=f.classify_features(f.features(y)).softmax(1);targets.append(p.argmax(1).cpu());conf.append(p.max(1).values.cpu())
X=torch.cat(features).double();label=torch.cat(targets);good=torch.cat(conf)>=.95;X=X[good];label=label[good];mean=X.mean(0);std=X.std(0,unbiased=False).clamp_min(.01);X=(X-mean)/std;X=torch.cat((X,torch.ones(len(X),1,dtype=X.dtype)),1);one=torch.nn.functional.one_hot(label,num_classes=6).double();reg=torch.eye(X.shape[1],dtype=X.dtype)*10.;reg[-1,-1]=1e-8;weights=torch.linalg.solve(X.T@X+reg,X.T@one)
if not torch.isfinite(weights).all():raise ValueError('Nonfinite source ridge probe')
received,scene=s.epoch_data(cache,14);d=prior.frozen_dura(root,'cpu');rows=[]
with torch.no_grad():
 for start in range(0,len(held),64):
  i=held[start:start+64];x=core.tensor(received[i],'cpu');y=core.tensor(clean[i],'cpu');cp=f.classify_features(f.features(y)).softmax(1);valid=cp.max(1).values>=.95;target=cp.argmax(1);feat=f.features(x);raw=f.classify_features(feat).argmax(1);z=(feat.double()-mean)/std;z=torch.cat((z,torch.ones(len(z),1,dtype=z.dtype)),1);pred=(z@weights).argmax(1);teacher=f.classify_features(f.features(d(x))).argmax(1);rows.append(torch.stack((valid.long(),(raw==target).long(),(pred==target).long(),(teacher==target).long(),((raw==target)|(pred==target)).long()),1).cpu())
v=np.from_dlpack(torch.cat(rows).contiguous());out={}
for name in s.VIEWS:
 g=v[scene[held]==name];g=g[g[:,0]>0];out[name]={'valid':len(g),'raw_F':int(g[:,1].sum()),'linear_probe':int(g[:,2].sum()),'DURA_F':int(g[:,3].sum()),'raw_probe_oracle':int(g[:,4].sum())}
print(json.dumps({'status':'SOURCE_ONLY_FEATURE_INFORMATION_DIAGNOSTIC','training_physical_ids':len(chosen),'valid_training_pseudo_targets':int(good.sum()),'heldout_physical_ids':len(held),'training_cache':7,'evaluation_cache':14,'F_feature_dim':160,'target':'high-confidence F(clean) pseudo class, threshold 0.95','method':'fixed L2-regularized linear least-squares probe on frozen F(received) features; lambda 10, no validation tuning','limitations':'probe predicts classes, does not produce IQ and is not a restoration model; source physical identities appeared in prior experiments; oracle uses pseudo target and is not deployable','TX_labels_read':0,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'results':out},allow_nan=False))
