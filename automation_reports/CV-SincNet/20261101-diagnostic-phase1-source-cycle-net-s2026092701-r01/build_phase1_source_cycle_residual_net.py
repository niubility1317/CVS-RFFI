"""Build a cycle-constrained, source-paired residual restoration network."""
import ast
from pathlib import Path

p = Path(__file__).parent
source = (p / 'phase1_source_cycle_projection_probe.py').read_text(encoding='utf8')
tree = ast.parse(source)
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
old = "split[np.isin(source_rx,[4,6])&(source_day==3)]=1"
new = "split[np.isin(source_rx,[1,3,4])&np.isin(source_day,[1,2])]=0;" + old
assert inner.count(old) == 1
inner = inner.replace(old, new)
assert inner.count('keep=split>0') == 1
inner = inner.replace('keep=split>0', 'keep=split>=0')
old = 'for view in views:\n if view=='
helper = r'''def align_to_anchor(wave,anchor):
 pad=np.pad(wave,((0,0),(16,16)),mode='reflect');target=anchor[:,16:-16];best=np.full(len(wave),np.inf);chosen=np.zeros_like(wave)
 for shift in range(-16,17):
  full=pad[:,16+shift:16+shift+256];part=full[:,16:-16];gain=np.sum(np.conj(part)*target,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-target)**2,axis=1);use=mse<best;best[use]=mse[use];chosen[use]=gain[use,None]*full[use]
 return chosen
def remove_cycle(z):
 mean=np.stack([np.mean(z[:,r::period],axis=1) for r in range(period)],1);return z-mean[:,residues]
'''
assert inner.count(old) == 1
inner = inner.replace(old, helper + old)
marker = 'for alpha in alphas:'
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + r'''
 received_complex=received[:,0].astype(np.float64)+1j*received[:,1];raw_aligned=align_to_anchor(received_complex,z);clean_aligned=align_to_anchor(clean_complex,z);rotation=np.exp(-1j*phi[:,None]*cycles);rms=np.sqrt(np.mean(abs(z)**2,axis=1)).clip(min=1e-8);canonical=z*rotation/rms[:,None];q_anchor=remove_cycle(canonical);q_raw=remove_cycle(raw_aligned*rotation/rms[:,None]);q_target=remove_cycle((clean_aligned-z)*rotation/rms[:,None]);features=np.stack((canonical.real,canonical.imag,q_anchor.real,q_anchor.imag,q_raw.real,q_raw.imag),1).astype(np.float32);features=np.pad(features,((0,0),(0,0),(0,4))).reshape(n,6,13,20);target_iq=np.stack((q_target.real,q_target.imag),1).astype(np.float32)
 data[view]=dict(input=core.tensor(features,'cpu'),target=core.tensor(target_iq,'cpu'),anchor=z,basep=basep,rawp=rawp,rotation=rotation,rms=rms,gate=gate);print('CYCLE_NET_SOURCE_INPUTS_FIXED',view,flush=True)
class CycleNet(torch.nn.Module):
 def __init__(self):
  super().__init__();self.net=torch.nn.Sequential(torch.nn.Conv2d(6,16,3,padding=1),torch.nn.GELU(),torch.nn.Conv2d(16,16,3,padding=1),torch.nn.GELU(),torch.nn.Conv2d(16,2,3,padding=1));torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
 def forward(self,x):
  out=self.net(x).reshape(-1,2,260)[:,:,:256];mean=torch.stack([out[:,:,r::20].mean(2) for r in range(20)],2);return out-mean[:,:,torch.arange(256,device=out.device)%20]
train_input=torch.cat([data[v]['input'] for v in views]);train_target=torch.cat([data[v]['target'] for v in views]);train_split=np.tile(split,len(views));train_gate=np.concatenate([data[v]['gate'] for v in views]);train=np.flatnonzero((train_split==0)&train_gate);assert len(train)>1000
torch.manual_seed(2026110421);model=CycleNet();opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001);rng=np.random.default_rng(2026110422);epochs=(2,5,10);states={};training_loss=[]
for epoch in range(1,11):
 model.train();order=rng.permutation(train);total=0.;num=0
 for start in range(0,len(order),64):
  idx=torch.tensor(order[start:start+64].tolist(),dtype=torch.long);pred=model(train_input[idx]);loss=(pred-train_target[idx]).square().mean()+.05*pred.square().mean()
  if not torch.isfinite(loss):raise FloatingPointError('finite cycle residual loss required')
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(loss)*len(idx);num+=len(idx)
 training_loss.append(total/num)
 if epoch in epochs:states[epoch]={k:v.detach().clone() for k,v in model.state_dict().items()};print('CYCLE_NET_SOURCE_EPOCH_FIXED',epoch,flush=True)
eval_ids=np.flatnonzero(split>0);alphas=(.25,.5,1.);predictions={};candidate_waves={}
for epoch in epochs:
 model.load_state_dict(states[epoch]);model.eval();delta={}
 with torch.no_grad():
  for view in views:
   pred=[];v=data[view]
   for start in range(0,len(eval_ids),128):
    ix=torch.tensor(eval_ids[start:start+128].tolist(),dtype=torch.long);pred.append(model(v['input'][ix]).cpu())
   wave=np.from_dlpack(torch.cat(pred).contiguous());delta[view]=(wave[:,0].astype(np.float64)+1j*wave[:,1])*v['rms'][eval_ids,None]/v['rotation'][eval_ids]
 for alpha in alphas:
  key=f'{epoch}:{alpha}';predictions[key]={};candidate_waves[key]={}
  for view in views:
   v=data[view];out=v['anchor'][eval_ids]+alpha*v['gate'][eval_ids,None]*delta[view];candidate_waves[key][view]=out;wiq=np.stack((out.real,out.imag),1).astype(np.float32);prob=[]
   with torch.no_grad():
    for start in range(0,len(eval_ids),64):
     x=core.tensor(wiq[start:start+64],'cpu');prob.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu())
   predictions[key][view]=np.from_dlpack(torch.cat(prob).contiguous())
 print('CYCLE_NET_SOURCE_CANDIDATES_FIXED',epoch,flush=True)
print('ALL_CYCLE_NET_SOURCE_CANDIDATES_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=data['clean']['rawp'];teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target)
def evaluate(key,part,group=None):
 mask=(split[eval_ids]==part)&valid[eval_ids];cm=split[eval_ids]==part
 if group is not None:mask &= source_rx[eval_ids]==group;cm &= source_rx[eval_ids]==group
 per={};clean_net=0
 for view in views:
  v=data[view];base=v['basep'][eval_ids].argmax(1);out=predictions[key][view].argmax(1);truth=target[eval_ids];old=base==truth;new=out==truth;lm=mask&labeled[eval_ids]
  per[view]=dict(valid_count=int(mask.sum()),L_s_count=int(lm.sum()),U_confident_count=int((mask&~labeled[eval_ids]).sum()),baseline=int((mask&old).sum()),student=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()))
  if view=='clean':clean_net=int((cm&(out==teacher_target[eval_ids])).sum()-(cm&(base==teacher_target[eval_ids])).sum())
 return per,clean_net
validation=[];best={'kind':'dura','validation_net':0}
for epoch in epochs:
 for alpha in alphas:
  key=f'{epoch}:{alpha}';per,clean_net=evaluate(key,1);changes={v:per[v]['student']-per[v]['baseline'] for v in views[1:]};net=sum(changes.values());groups_v={}
  for group in (4,6):
   gp,gclean=evaluate(key,1,group);gc={v:gp[v]['student']-gp[v]['baseline'] for v in views[1:]};groups_v[str(group)]=dict(n=int(((split==1)&(source_rx==group)).sum()),net=sum(gc.values()),per_view=gc,clean_teacher_net=gclean)
  validation.append(dict(epoch=epoch,alpha=alpha,net=net,per_view=changes,clean_teacher_net=clean_net,groups=groups_v))
  guards=all(z['net']>=0 and z['clean_teacher_net']>=0 and min(z['per_view'].values())>=-.005*z['n'] for z in groups_v.values())
  if net>best['validation_net'] and guards:best=dict(kind='cycle_residual_net',epoch=epoch,alpha=alpha,validation_net=net)
def aligned_mse(wave,truth):
 pad=np.pad(wave,((0,0),(16,16)),mode='reflect');truth=truth[:,16:-16];best=np.full(len(wave),np.inf)
 for shift in range(-16,17):
  part=pad[:,16+shift:16+shift+256][:,16:-16];gain=np.sum(np.conj(part)*truth,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-truth)**2,axis=1);best=np.minimum(best,mse)
 return best
if best['kind']=='dura':heldout=None;heldout_clean=None;quality=None;state=None
else:
 key=f"{best['epoch']}:{best['alpha']}";heldout,heldout_clean=evaluate(key,2);select=split[eval_ids]==2;truth=clean_complex[eval_ids][select];quality={}
 for view in views:
  before=aligned_mse(data[view]['anchor'][eval_ids][select],truth);after=aligned_mse(candidate_waves[key][view][select],truth);quality[view]=dict(count=int(select.sum()),aligned_mse_before=float(before.mean()),aligned_mse_after=float(after.mean()),improved=int((after<before).sum()),worsened=int((after>before).sum()))
 state={k:v.tolist() for k,v in states[best['epoch']].items()}
print('SOURCE_CYCLE_NET '+json.dumps(dict(status='SOURCE_ONLY_CYCLE_RESIDUAL_NET',source_ids=n,train_ids=int((split==0).sum()),validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),training_records=int(len(train)),training_loss=training_loss,validation=validation,selected=best,heldout=heldout,heldout_clean_teacher_net=heldout_clean,quality=quality,model_state=state,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,candidates_before_source_tx_labels=True,
 caveat='Period and architecture were inspired by earlier source observations; source IDs overlap those cycle diagnostics. The regressor uses source clean IQ aligned to DURA as a training target, never an inference input. Actual received IQ is aligned to DURA without reference to clean IQ at inference. Only selector/regressor receiver-day is held out; frozen F and DURA saw source data. Simulator not validated real satellite. Earlier Phase1 test observations motivated exploration.'),ensure_ascii=False),flush=True)
'''
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=3600)
'''.replace('INNER', repr(inner), 1)
wrapper = '''import json,subprocess
from pathlib import Path
remote=REMOTE
proc=subprocess.Popen(['C:/Program Files/Git/usr/bin/ssh.exe','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607','python3','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',bufsize=1)
proc.stdin.write(remote);proc.stdin.close();marker='SOURCE_CYCLE_NET ';payload=None
for line in proc.stdout:
 if line.startswith(marker):payload=line[len(marker):]
 else:print(line,end='',flush=True)
rc=proc.wait()
if rc==0:
 if payload is None:raise ValueError('source result marker missing')
 dest=Path(__file__).parent/'phase1_source_cycle_residual_net.json'
 if dest.exists():assert json.loads(dest.read_text(encoding='utf8'))==json.loads(payload)
 else:dest.write_text(payload,encoding='utf8',newline='\\n')
 print('SAVED_SOURCE_CYCLE_NET',dest,flush=True)
raise SystemExit(rc)
'''.replace('REMOTE', repr(remote), 1)
dest = p / 'phase1_source_cycle_residual_net.py'
if dest.exists():
    assert 'SOURCE_CYCLE_NET' in dest.read_text(encoding='utf8')
compile(inner, '<cycle-net-inner>', 'exec')
compile(wrapper, str(dest), 'exec')
dest.write_text(wrapper, encoding='utf8', newline='\n')
print('CREATED source cycle residual network', dest)
