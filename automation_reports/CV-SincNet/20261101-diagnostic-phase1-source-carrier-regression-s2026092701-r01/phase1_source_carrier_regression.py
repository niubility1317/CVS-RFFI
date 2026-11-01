"""Source-only physical carrier-phase regression; no TX labels in restoration."""
import json
import subprocess
from pathlib import Path

inner = r'''
from pathlib import Path
import json,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,physical_rich_router_experiment as g,feature_conditioned_experiment as fs
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
models=g.load_models(root,'cpu');teacher,contract,dura,*_=models;cache,clean,roles=core.data(root,contract)
from leo_practical.channel import Config,receiver_for_session,stable_seed,normalize_input
from leo_practical.residual import ResidualChannel
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);rx=np.load(cache/'train/rx.npy',allow_pickle=False);day=np.load(cache/'train/day.npy',allow_pickle=False)
q_rng=np.random.default_rng(2026110251);h_rng=np.random.default_rng(2026110351);guard_rng=np.random.default_rng(2026110361);rng=np.random.default_rng(2026110371);groups=[]
for role,q_count,h_count,guard_count,take in (('L_s',2048,512,2048,1024),('U_s',6144,1536,6144,3072)):
 available=np.flatnonzero(roles==role);q_ids=q_rng.choice(available,size=q_count,replace=False);h_ids=h_rng.choice(available,size=h_count,replace=False)
 without_h=np.setdiff1d(available,np.sort(h_ids),assume_unique=True);guard_ids=guard_rng.choice(without_h,size=guard_count,replace=False);excluded=np.unique(np.concatenate((q_ids,h_ids,guard_ids)));available=np.setdiff1d(available,excluded,assume_unique=True);assert len(available)>=take
 groups.append(np.sort(rng.choice(available,size=take,replace=False)))
fixed=np.concatenate(groups);assert len(fixed)==len(np.unique(fixed))==4096
source_rx=rx[fixed];source_day=day[fixed];split=np.full(len(fixed),-1,dtype=np.int64);split[np.isin(source_rx,[1,3,4])&np.isin(source_day,[1,2])]=0;split[np.isin(source_rx,[4,6])&(source_day==3)]=1;split[(source_rx==8)&(source_day==3)]=2
assert all(int((split==i).sum())>200 for i in range(3));n=len(fixed);views=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban');data={};namespace='phase1-source-carrier-2026110391'
clean_wave=core.tensor(np.asarray(clean[fixed],dtype=np.float32),'cpu');zc=torch.complex(clean_wave[:,0],clean_wave[:,1])
with torch.no_grad():
 for view in views:
  if view=='clean':received=np.asarray(clean[fixed],dtype=np.float32)
  else:
   cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');received=[]
   for j in fixed:
    pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110392,session);v=np.asarray(clean[j],dtype=np.float64);signal,_=normalize_input(v[0]+1j*v[1]);result,meta=ResidualChannel(cfg,stable_seed(2026110391,namespace,view,pid),hw).process(signal)
    assert np.isfinite(result).all() and not meta['channel_equalization_applied'];received.append(np.stack((result.real,result.imag)).astype(np.float32))
   received=np.stack(received)
  inp=core.tensor(received,'cpu');anchors=[];rawp=[];basep=[]
  for start in range(0,n,32):
   x=inp[start:start+32];a=dura(x);anchors.append(a.cpu());rawp.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu());basep.append(F.softmax(teacher.classify_features(teacher.features(a)),1).cpu())
  anchor=torch.cat(anchors).contiguous();rms=inp.square().sum(1).mean(1).sqrt().clamp_min(1e-8);ratio=((inp-anchor).square().sum((1,2))/inp.square().sum((1,2)).clamp_min(1e-12)).sqrt();za=torch.complex(anchor[:,0],anchor[:,1]);corr=(za.conj()*zc).sum(1);weight=(corr.abs()/(za.abs().square().sum(1)*zc.abs().square().sum(1)).sqrt().clamp_min(1e-12)).clamp(0,1);direction=corr/corr.abs().clamp_min(1e-12)
  data[view]=dict(input=torch.cat((inp/rms[:,None,None],anchor/rms[:,None,None]),1),anchor=anchor,ratio=ratio,target=torch.stack((direction.real,direction.imag),1),weight=weight,rawp=torch.cat(rawp).contiguous(),basep=torch.cat(basep).contiguous())
  print('CARRIER_SOURCE_BASELINES_FIXED',view,flush=True)
class PhaseNet(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(4,32,7,stride=2,padding=3),nn.GELU(),nn.Conv1d(32,64,5,stride=2,padding=2),nn.GELU(),nn.Conv1d(64,96,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool1d(1));self.head=nn.Sequential(nn.Linear(96,96),nn.GELU(),nn.Linear(96,2))
 def forward(self,x):return self.head(self.net(x).squeeze(-1))
all_input=torch.cat([data[v]['input'] for v in views]);all_target=torch.cat([data[v]['target'] for v in views]);all_weight=torch.cat([data[v]['weight'] for v in views]);all_split=np.tile(split,len(views));train=np.flatnonzero(all_split==0);assert len(train)>1000
torch.manual_seed(2026110393);model=PhaseNet();opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001);rng=np.random.default_rng(2026110394);epochs=(2,5,10,20);thresholds=(.1,.3,.6);phase_est={};states={};train_loss=[]
for epoch in range(1,21):
 model.train();order=rng.permutation(train);total=0.;denom=0.
 for start in range(0,len(order),128):
  ix=torch.tensor(order[start:start+128].tolist(),dtype=torch.long);pred=model(all_input[ix]);w=all_weight[ix];loss=(((pred-all_target[ix])**2).sum(1)*w).sum()/w.sum().clamp_min(1e-8)
  if not torch.isfinite(loss):raise FloatingPointError('finite carrier regression loss required')
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(loss)*len(ix);denom+=len(ix)
 train_loss.append(total/denom)
 if epoch in epochs:
  model.eval();est=[]
  with torch.no_grad():
   for start in range(0,len(all_input),256):est.append(model(all_input[start:start+256]).cpu())
  phase_est[epoch]=torch.cat(est).reshape(len(views),n,2);states[epoch]={k:v.detach().clone() for k,v in model.state_dict().items()};print('CARRIER_SOURCE_PHASE_ESTIMATE_FIXED',epoch,flush=True)
evaluation=np.flatnonzero(split>0);predictions={};quality={}
with torch.no_grad():
 for epoch in epochs:
  for threshold in thresholds:
   key=f'{epoch}:{threshold}';predictions[key]={};quality[key]={}
   for vi,view in enumerate(views):
    v=data[view];est=phase_est[epoch][vi];theta=torch.atan2(est[:,1],est[:,0]);confidence=est.square().sum(1).sqrt();apply=(confidence>=threshold)&(v['ratio']>.2);theta=torch.where(apply,theta,torch.zeros_like(theta));pred=[]
    for start in range(0,len(evaluation),64):
     idx=torch.tensor(evaluation[start:start+64].tolist(),dtype=torch.long);a=v['anchor'][idx];z=torch.complex(a[:,0],a[:,1])*torch.exp(1j*theta[idx,None]);wave=torch.stack((z.real,z.imag),1);pred.append(F.softmax(teacher.classify_features(teacher.features(wave)),1).cpu())
    predictions[key][view]=np.from_dlpack(torch.cat(pred).contiguous());target_theta=torch.atan2(v['target'][:,1],v['target'][:,0]);quality[key][view]=dict(correlation_weighted_before=float(((1-torch.cos(target_theta))*v['weight'])[torch.tensor(evaluation.tolist())].sum()/v['weight'][torch.tensor(evaluation.tolist())].sum().clamp_min(1e-8)),correlation_weighted_after=float(((1-torch.cos(theta-target_theta))*v['weight'])[torch.tensor(evaluation.tolist())].sum()/v['weight'][torch.tensor(evaluation.tolist())].sum().clamp_min(1e-8)),applied=int(apply[torch.tensor(evaluation.tolist())].sum()))
 print('CARRIER_SOURCE_CANDIDATES_FIXED',epoch,flush=True)
print('ALL_CARRIER_SOURCE_CANDIDATES_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=np.from_dlpack(data['clean']['rawp']);teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target);baselines={v:np.from_dlpack(data[v]['basep']).argmax(1) for v in views}
def evaluate(key,part,group=None):
 mask=(split[evaluation]==part)&valid[evaluation];cm=split[evaluation]==part
 if group is not None:mask &= source_rx[evaluation]==group;cm &= source_rx[evaluation]==group
 per={};clean_net=0
 for view in views:
  base=baselines[view][evaluation];out=predictions[key][view].argmax(1);truth=target[evaluation];lm=mask&labeled[evaluation];old=base==truth;new=out==truth
  per[view]=dict(valid_count=int(mask.sum()),L_s_count=int(lm.sum()),U_confident_count=int((mask&~labeled[evaluation]).sum()),baseline=int((mask&old).sum()),student=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()))
  if view=='clean':clean_net=int((cm&(out==teacher_target[evaluation])).sum()-(cm&(base==teacher_target[evaluation])).sum())
 return per,clean_net
validation=[];best={'kind':'dura','validation_net':0}
for epoch in epochs:
 for threshold in thresholds:
  key=f'{epoch}:{threshold}';per,clean_net=evaluate(key,1);changes={v:per[v]['student']-per[v]['baseline'] for v in views[1:]};net=sum(changes.values());groups_v={}
  for group in (4,6):
   gp,gclean=evaluate(key,1,group);gc={v:gp[v]['student']-gp[v]['baseline'] for v in views[1:]};groups_v[str(group)]=dict(net=sum(gc.values()),per_view=gc,clean_teacher_net=gclean)
  validation.append(dict(epoch=epoch,threshold=threshold,net=net,per_view=changes,clean_teacher_net=clean_net,groups=groups_v))
  guards=all(z['net']>=0 and z['clean_teacher_net']>=0 and min(z['per_view'].values())>=0 for z in groups_v.values())
  if net>best['validation_net'] and guards:best=dict(kind='carrier_phase_regressor',epoch=epoch,threshold=threshold,validation_net=net)
if best['kind']=='dura':heldout=None;heldout_clean=None;state=None;selected_quality=None
else:
 key=f"{best['epoch']}:{best['threshold']}";heldout,heldout_clean=evaluate(key,2);state={k:v.tolist() for k,v in states[best['epoch']].items()};selected_quality=quality[key]
print('SOURCE_CARRIER_REGRESSION '+json.dumps(dict(status='SOURCE_ONLY_CARRIER_REGRESSION',source_ids=n,train_ids=int((split==0).sum()),validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),training_loss=train_loss,validation=validation,selected=best,heldout=heldout,heldout_clean_teacher_net=heldout_clean,selected_quality_all_validation_and_heldout=selected_quality,model_state=state,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,candidates_before_source_tx_labels=True,
 caveat='Source clean waveform is a training and quality reference, not an inference input. Source IDs overlap earlier selector experiments, so this is exploratory. Only the regressor uses receiver/day holdout; the frozen models saw source data. Simulator is not validated real satellite data; previously observed Phase1 development-test results motivated exploration.'),ensure_ascii=False),flush=True)
'''
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=3600)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.Popen(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'N607', 'python3', '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf8', errors='replace', bufsize=1)
proc.stdin.write(remote)
proc.stdin.close()
marker = 'SOURCE_CARRIER_REGRESSION '
payload = None
for line in proc.stdout:
    if line.startswith(marker):
        payload = line[len(marker):]
    else:
        print(line, end='', flush=True)
rc = proc.wait()
if rc == 0:
    if payload is None:
        raise ValueError('source result marker missing')
    dest = Path(__file__).parent / 'phase1_source_carrier_regression.json'
    if dest.exists():
        assert json.loads(dest.read_text(encoding='utf8')) == json.loads(payload)
    else:
        dest.write_text(payload, encoding='utf8', newline='\n')
    print('SAVED_SOURCE_CARRIER_REGRESSION', dest, flush=True)
raise SystemExit(rc)
