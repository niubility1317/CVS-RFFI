"""Source-only learned cycle spectral shrinkage with frozen ground teacher."""
import ast,json,subprocess,sys
from pathlib import Path
p=Path(__file__).parent
tree=ast.parse((p/'phase1_source_cycle_lowrank_probe.py').read_text(encoding='utf8'))
inner=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='inner' for t in n.targets)))
inner=inner.replace('2026110521','2026110531').replace('2026110522','2026110532').replace('2026110523','2026110533')
assert inner.count('keep=split>0')==1
inner=inner.replace('keep=split>0','split[np.isin(source_rx,[1,3,4])&np.isin(source_day,[1,2])]=0;keep=split>=0')
if '--smoke' in sys.argv:
 old='n=len(fixed);assert n>300';new=old+";sel=np.concatenate((np.flatnonzero(split==0)[:32],np.flatnonzero(source_rx==4)[-4:],np.flatnonzero(source_rx==8)[:4]));fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel];n=len(fixed)";assert inner.count(old)==1;inner=inner.replace(old,new)
marker="for view in views:\n if view=='clean':received=clean_iq"
assert inner.count(marker)==1;inner=inner.split(marker,1)[0]
inner+=r'''
def align_clean_to_anchor(clean,anchor):
 pad=np.pad(clean,((0,0),(16,16)),mode='reflect');ref=anchor[:,16:-16];best=np.full(len(clean),np.inf);out=np.zeros_like(clean)
 for shift in range(-16,17):
  part=pad[:,16+shift:16+shift+256];crop=part[:,16:-16];gain=np.sum(np.conj(crop)*ref,axis=1)/np.maximum(np.sum(abs(crop)**2,axis=1),1e-20);err=np.mean(abs(gain[:,None]*crop-ref)**2,axis=1);use=err<best;best[use]=err[use];out[use]=gain[use,None]*part[use]
 return out
teacher.eval();teacher.requires_grad_(False);ground_features=[];ground_probs=[]
with torch.no_grad():
 for start in range(0,n,64):
  gt=core.tensor(clean_iq[start:start+64],'cpu');f=teacher.features(gt);ground_features.append(F.normalize(f,dim=1).cpu());ground_probs.append(F.softmax(teacher.classify_features(f),dim=1).cpu())
ground_features=torch.cat(ground_features).contiguous();ground_probs=torch.cat(ground_probs).contiguous()
for view in views:
 if view=='clean':received=clean_iq
 else:
  cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');received=[]
  for j in fixed:
   pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110533,session);v=np.asarray(clean[j],dtype=np.float64);sig,_=normalize_input(v[0]+1j*v[1]);out,meta=ResidualChannel(cfg,stable_seed(2026110532,'source-cycle-spectral-network-2026110532',view,pid),hw).process(sig);assert np.isfinite(out).all() and not meta['channel_equalization_applied'];received.append(np.stack((out.real,out.imag)).astype(np.float32))
  received=np.stack(received)
 anchors=[];rawp=[];basep=[]
 with torch.no_grad():
  for start in range(0,n,32):
   x=core.tensor(received[start:start+32],'cpu');a=dura(x);anchors.append(a.cpu());rawp.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu());basep.append(F.softmax(teacher.classify_features(teacher.features(a)),1).cpu())
 anchor=np.from_dlpack(torch.cat(anchors).contiguous());rawp=np.from_dlpack(torch.cat(rawp).contiguous());basep=np.from_dlpack(torch.cat(basep).contiguous());z=anchor[:,0].astype(np.float64)+1j*anchor[:,1];raw=received[:,0].astype(np.float64)+1j*received[:,1];_,phi=project(z);rotation=np.exp(-1j*phi[:,None]*cycles);demod=z*rotation;matrix=demod[:,:240].reshape(n,12,20);q=matrix-matrix.mean(1,keepdims=True);u,s,vh=np.linalg.svd(q,full_matrices=False);piece=(u.transpose(0,2,1)[:,:,:,None]*s[:,:,None,None]*vh[:,:,None,:]).reshape(n,12,240);basis=np.zeros((n,12,256),dtype=np.complex128);basis[:,:,:240]=piece;basis/=rotation[:,None,:];basis_iq=np.stack((basis.real,basis.imag),2).astype(np.float32);assert np.max(abs(basis.sum(1)[:,:240]*rotation[:,:240]-q.reshape(n,240)))<1e-8
 ratio=np.sqrt(np.sum((received-anchor)**2,axis=(1,2))/np.maximum(np.sum(received**2,axis=(1,2)),1e-12));power=np.mean(abs(raw[:,64:-64])**2,axis=1).clip(min=1e-20);fraction=np.clip(wide_mse(z,raw)/power,1e-8,1.);proxy=10*np.log10(np.maximum(1-fraction,1e-8)/fraction);gate=(ratio>.2)&(proxy<10.);features=np.concatenate((np.log(np.maximum(s,1e-8)/np.maximum(s[:,:1],1e-8)),np.clip(proxy[:,None],-20,30)/20,np.clip(ratio[:,None],0,3)),1).astype(np.float32);target=align_clean_to_anchor(clean_complex,z);target_iq=np.stack((target.real,target.imag),1).astype(np.float32)
 data[view]=dict(anchor=core.tensor(anchor.copy(),'cpu'),basis=core.tensor(basis_iq,'cpu'),features=core.tensor(features,'cpu'),target=core.tensor(target_iq,'cpu'),gate=gate,rawp=rawp,basep=basep);print('SPECTRAL_NETWORK_SOURCE_INPUTS_FIXED',view,flush=True)
class SpectralGate(torch.nn.Module):
 def __init__(self):
  super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(14,48),torch.nn.GELU(),torch.nn.Linear(48,12));torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.constant_(self.net[-1].bias,-4.)
 def forward(self,x):return .75*torch.sigmoid(self.net(x))
torch.manual_seed(2026110534);model=SpectralGate();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);rng=np.random.default_rng(2026110535)
train_inputs=torch.cat([data[v]['features'] for v in views[1:]]);train_basis=torch.cat([data[v]['basis'] for v in views[1:]]);train_anchor=torch.cat([data[v]['anchor'] for v in views[1:]]);train_target=torch.cat([data[v]['target'] for v in views[1:]]);train_gtf=ground_features.repeat(3,1);train_gtp=ground_probs.repeat(3,1);train_split=np.tile(split,3);train_gate=np.concatenate([data[v]['gate'] for v in views[1:]]);train=np.flatnonzero((train_split==0)&train_gate);assert len(train)>10
states={};losses=[]
for epoch in range(1,11):
 model.train();order=rng.permutation(train);total=0.;num=0
 for start in range(0,len(order),64):
  ix=torch.tensor(order[start:start+64].tolist(),dtype=torch.long);shrink=model(train_inputs[ix]);wave=train_anchor[ix]-torch.einsum('bk,bkct->bct',shrink,train_basis[ix]);features=teacher.features(wave);f=F.normalize(features,dim=1);fidelity=(wave-train_target[ix]).square().mean((1,2))/train_target[ix].square().mean((1,2)).clamp_min(1e-8);semantic=1-(f*train_gtf[ix]).sum(1);logp=F.log_softmax(teacher.classify_features(features),1);gp=train_gtp[ix];kl=(gp*(gp.clamp_min(1e-8).log()-logp)).sum(1);loss=(fidelity+.1*semantic+.1*kl+.02*shrink.square().mean(1)).mean()
  if not torch.isfinite(loss):raise FloatingPointError('finite spectral training loss required')
  optimizer.zero_grad(set_to_none=True);loss.backward();assert model.net[-1].weight.grad is not None and torch.isfinite(model.net[-1].weight.grad).all() and float(model.net[-1].weight.grad.abs().sum())>0;torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();total+=float(loss)*len(ix);num+=len(ix)
 losses.append(total/num)
 if epoch in (2,5,10):states[epoch]={k:v.detach().clone() for k,v in model.state_dict().items()};print('SPECTRAL_NETWORK_EPOCH_FIXED',epoch,flush=True)
assert all(p.grad is None for p in teacher.parameters());eval_ids=np.flatnonzero(split>0);alphas=(.25,.5,1.);preds={};deltas={}
for epoch in (2,5,10):
 model.load_state_dict(states[epoch]);model.eval()
 with torch.no_grad():
  for view in views:
   d=data[view];parts=[]
   for start in range(0,len(eval_ids),64):
    ix=torch.tensor(eval_ids[start:start+64].tolist(),dtype=torch.long);g=model(d['features'][ix]);parts.append(torch.einsum('bk,bkct->bct',g,d['basis'][ix]).cpu())
   delta=torch.cat(parts).contiguous();deltas[epoch,view]=delta
   for alpha in alphas:
    outputs=[]
    for start in range(0,len(eval_ids),64):
     s0=slice(start,start+64);ix=torch.tensor(eval_ids[s0].tolist(),dtype=torch.long);gate_t=torch.tensor(d['gate'][eval_ids[s0]].astype(np.float32).tolist())[:,None,None];wave=d['anchor'][ix]-alpha*gate_t*delta[s0];outputs.append(F.softmax(teacher.classify_features(teacher.features(wave)),1).cpu())
    preds[epoch,alpha,view]=np.from_dlpack(torch.cat(outputs).contiguous())
 print('SPECTRAL_NETWORK_CANDIDATES_FIXED',epoch,flush=True)
print('ALL_SPECTRAL_NETWORK_CANDIDATES_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=data['clean']['rawp'];teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target)
def evaluate(epoch,alpha,part,group=None):
 mask=(split[eval_ids]==part)&valid[eval_ids];cm=split[eval_ids]==part
 if group is not None:mask &= source_rx[eval_ids]==group;cm &= source_rx[eval_ids]==group
 out={};clean_net=0
 for view in views:
  base=data[view]['basep'][eval_ids].argmax(1);pred=preds[epoch,alpha,view].argmax(1);truth=target[eval_ids];old=base==truth;new=pred==truth;lm=mask&labeled[eval_ids]
  out[view]=dict(valid_count=int(mask.sum()),L_s_count=int(lm.sum()),baseline=int((mask&old).sum()),student=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()))
  if view=='clean':clean_net=int((cm&(pred==teacher_target[eval_ids])).sum()-(cm&(base==teacher_target[eval_ids])).sum())
 return out,clean_net
validation=[];best={'kind':'dura','validation_net':0}
for epoch in (2,5,10):
 for alpha in alphas:
  per,clean_net=evaluate(epoch,alpha,1);changes={v:per[v]['student']-per[v]['baseline'] for v in views[1:]};net=sum(changes.values());groups={}
  for group in (4,6):
   gp,gclean=evaluate(epoch,alpha,1,group);gc={v:gp[v]['student']-gp[v]['baseline'] for v in views[1:]};groups[str(group)]=dict(n=int(((split==1)&(source_rx==group)).sum()),net=sum(gc.values()),per_view=gc,clean_teacher_net=gclean)
  validation.append(dict(epoch=epoch,alpha=alpha,net=net,per_view=changes,clean_teacher_net=clean_net,groups=groups));guards=all(z['net']>=0 and z['clean_teacher_net']>=0 and min(z['per_view'].values())>=-.005*z['n'] for z in groups.values())
  if net>best['validation_net'] and guards:best=dict(kind='spectral_shrinkage_network',epoch=epoch,alpha=alpha,validation_net=net)
if best['kind']=='dura':heldout=None;quality=None;state=None
else:
 heldout,_=evaluate(best['epoch'],best['alpha'],2);mask=split[eval_ids]==2;quality={}
 for view in views:
  d=data[view];ix=torch.tensor(eval_ids.tolist(),dtype=torch.long);gate=torch.tensor(d['gate'][eval_ids].astype(np.float32).tolist())[:,None,None];out=np.from_dlpack((d['anchor'][ix]-best['alpha']*gate*deltas[best['epoch'],view]).contiguous());base=np.from_dlpack(d['anchor'][ix].contiguous());b=base[:,0].astype(np.float64)+1j*base[:,1];w=out[:,0].astype(np.float64)+1j*out[:,1];truth=clean_complex[eval_ids];before=wide_mse(b[mask],truth[mask]);after=wide_mse(w[mask],truth[mask]);quality[view]=dict(count=int(mask.sum()),aligned_mse_before=float(before.mean()),aligned_mse_after=float(after.mean()),improved=int((after<before).sum()),worsened=int((after>before).sum()))
 state={k:v.detach().tolist() for k,v in states[best['epoch']].items()}
print('SOURCE_SPECTRAL_SHRINKAGE_NETWORK '+json.dumps(dict(status='SOURCE_ONLY_SPECTRAL_SHRINKAGE_NETWORK',source_ids=n,train_ids=int((split==0).sum()),validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),training_gated_views=int(len(train)),training_loss=losses,validation=validation,selected=best,heldout=heldout,quality=quality,model_state=state,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,predictions_before_source_labels=True,caveat='The network gates fixed SVD components in the demodulated 20-sample cycle residual. Training uses paired source clean IQ aligned to DURA plus unaligned clean frozen-teacher features and soft class targets; inference uses received IQ and fixed models only. Frozen teacher and DURA saw source records. ±64 source IQ error is posthoc; simulator is not validated real satellite data.'),ensure_ascii=False),flush=True)
'''
compile(inner,'<source-spectral-shrinkage-network>','exec')
remote='''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=3600)
'''.replace('INNER',repr(inner),1)
proc=subprocess.Popen(['C:/Program Files/Git/usr/bin/ssh.exe','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607','python3','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',bufsize=1)
proc.stdin.write(remote);proc.stdin.close();payload=None
for line in proc.stdout:
 if line.startswith('SOURCE_SPECTRAL_SHRINKAGE_NETWORK '):payload=line[len('SOURCE_SPECTRAL_SHRINKAGE_NETWORK '):]
 else:print(line,end='',flush=True)
rc=proc.wait()
if rc:raise SystemExit(rc)
if payload is None:raise ValueError('missing spectral network result')
res=json.loads(payload);assert res['U_s_labels_read']==res['V_reads']==res['test_IQ_reads']==0 and res['predictions_before_source_labels']
if '--smoke' in sys.argv:print('VERIFIED_SOURCE_SPECTRAL_NETWORK_SMOKE',res['source_ids']);raise SystemExit(0)
dest=p/'phase1_source_spectral_shrinkage_network.json'
if dest.exists():assert json.loads(dest.read_text(encoding='utf8'))==res
else:dest.write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
print('SAVED_SOURCE_SPECTRAL_NETWORK',dest,flush=True)
