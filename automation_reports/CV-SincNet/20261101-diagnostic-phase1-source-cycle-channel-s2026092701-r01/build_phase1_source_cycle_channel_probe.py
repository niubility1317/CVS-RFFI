"""Build per-record periodic pseudo-pilot channel inversion source pilot."""
import ast
from pathlib import Path

p = Path(__file__).parent
source = (p / 'phase1_source_cycle_projection_probe.py').read_text(encoding='utf8')
tree = ast.parse(source)
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
inner = inner.replace('2026110413', '2026110435').replace('2026110412', '2026110434').replace('source-cycle-projection-2026110434', 'source-cycle-channel-2026110434')
old = 'for view in views:\n if view=='
helper = r'''def align_to_anchor(wave,anchor):
 pad=np.pad(wave,((0,0),(16,16)),mode='reflect');target=anchor[:,16:-16];best=np.full(len(wave),np.inf);chosen=np.zeros_like(wave)
 for shift in range(-16,17):
  full=pad[:,16+shift:16+shift+256];part=full[:,16:-16];gain=np.sum(np.conj(part)*target,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-target)**2,axis=1);use=mse<best;best[use]=mse[use];chosen[use]=gain[use,None]*full[use]
 return chosen
def channel_from_cycles(wave,anchor):
 raw=align_to_anchor(wave,anchor);phi_a=np.angle(np.sum(np.conj(anchor[:,:-20])*anchor[:,20:],axis=1));phi_x=np.angle(np.sum(np.conj(raw[:,:-20])*raw[:,20:],axis=1));delta=np.angle(np.exp(1j*(phi_x-phi_a)));time=np.arange(256);raw=raw*np.exp(-1j*delta[:,None]*time/20);rotation=np.exp(-1j*phi_a[:,None]*time/20);a=anchor*rotation;x=raw*rotation;pa=np.stack([np.mean(a[:,r::20],axis=1) for r in range(20)],1);px=np.stack([np.mean(x[:,r::20],axis=1) for r in range(20)],1);A=np.fft.fft(pa,axis=1);X=np.fft.fft(px,axis=1);power=abs(A)**2;ridge=.05*np.maximum(power.max(1),1e-20);h=(np.conj(A)*X+ridge[:,None])/(power+ridge[:,None]);f20=np.fft.fftfreq(20);f256=np.fft.fftfreq(256);H=np.stack([np.interp(f256,f20,v.real,period=1)+1j*np.interp(f256,f20,v.imag,period=1) for v in h]);assert np.isfinite(H).all();return x,a,H,rotation
def invert(x,a,H,rotation,regularizer):
 X=np.fft.fft(x,axis=1);A=np.fft.fft(a,axis=1);y=np.fft.ifft((np.conj(H)*X+regularizer*A)/(abs(H)**2+regularizer),axis=1);return y/rotation
'''
assert inner.count(old) == 1
inner = inner.replace(old, helper + old)
marker = 'for alpha in alphas:'
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + r'''
 raw=received[:,0].astype(np.float64)+1j*received[:,1];x0,a0,H,rotation=channel_from_cycles(raw,z);regularizers=(.3,1.,3.,10.);blends=(.25,.5,1.);candidates=[];probs=[]
 for reg in regularizers:
  restored=invert(x0,a0,H,rotation,reg)
  for blend in blends:
   candidate=z+blend*gate[:,None]*(restored-z);candidate_iq=np.stack((candidate.real,candidate.imag),1).astype(np.float32);candidates.append(candidate);out=[]
   with torch.no_grad():
    for start in range(0,n,64):
     x=core.tensor(candidate_iq[start:start+64],'cpu');out.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu())
   probs.append(np.from_dlpack(torch.cat(out).contiguous()))
 data[view]=dict(anchor=z,rawp=rawp,basep=basep,probs=np.stack(probs),candidates=np.stack(candidates),gate=gate);print('CYCLE_CHANNEL_SOURCE_CANDIDATES_FIXED',view,flush=True)
print('ALL_CYCLE_CHANNEL_CANDIDATES_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=data['clean']['rawp'];teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target)
def evaluate(candidate,part,group=None):
 mask=(split==part)&valid;cm=split==part
 if group is not None:mask &= source_rx==group;cm &= source_rx==group
 per={};clean_net=0
 for view in views:
  v=data[view];base=v['basep'].argmax(1);out=v['probs'][candidate].argmax(1);old=base==target;new=out==target;lm=mask&labeled
  per[view]=dict(valid_count=int(mask.sum()),L_s_count=int(lm.sum()),U_confident_count=int((mask&~labeled).sum()),baseline=int((mask&old).sum()),student=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()))
  if view=='clean':clean_net=int((cm&(out==teacher_target)).sum()-(cm&(base==teacher_target)).sum())
 return per,clean_net
validation=[];best={'kind':'dura','validation_net':0}
for i in range(12):
 reg=regularizers[i//3];blend=blends[i%3];per,clean_net=evaluate(i,1);changes={v:per[v]['student']-per[v]['baseline'] for v in views[1:]};net=sum(changes.values());groups_v={}
 for group in (4,6):
  gp,gclean=evaluate(i,1,group);gc={v:gp[v]['student']-gp[v]['baseline'] for v in views[1:]};groups_v[str(group)]=dict(n=int(((split==1)&(source_rx==group)).sum()),net=sum(gc.values()),per_view=gc,clean_teacher_net=gclean)
 validation.append(dict(candidate=i,regularizer=reg,blend=blend,net=net,per_view=changes,clean_teacher_net=clean_net,groups=groups_v))
 guards=all(z['net']>=0 and z['clean_teacher_net']>=0 and min(z['per_view'].values())>=-.005*z['n'] for z in groups_v.values())
 if net>best['validation_net'] and guards:best=dict(kind='cycle_channel_inversion',candidate=i,regularizer=reg,blend=blend,validation_net=net)
def aligned_mse(wave,truth):
 pad=np.pad(wave,((0,0),(16,16)),mode='reflect');truth=truth[:,16:-16];best=np.full(len(wave),np.inf)
 for shift in range(-16,17):
  part=pad[:,16+shift:16+shift+256][:,16:-16];gain=np.sum(np.conj(part)*truth,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-truth)**2,axis=1);best=np.minimum(best,mse)
 return best
if best['kind']=='dura':heldout=None;heldout_clean=None;quality=None
else:
 heldout,heldout_clean=evaluate(best['candidate'],2);mask=split==2;quality={}
 for view in views:
  v=data[view];before=aligned_mse(v['anchor'][mask],clean_complex[mask]);after=aligned_mse(v['candidates'][best['candidate']][mask],clean_complex[mask]);quality[view]=dict(count=int(mask.sum()),aligned_mse_before=float(before.mean()),aligned_mse_after=float(after.mean()),improved=int((after<before).sum()),worsened=int((after>before).sum()))
print('SOURCE_CYCLE_CHANNEL '+json.dumps(dict(status='SOURCE_ONLY_CYCLE_CHANNEL_PROBE',source_ids=n,validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),period=20,regularizers=regularizers,blends=blends,validation=validation,selected=best,heldout=heldout,heldout_clean_teacher_net=heldout_clean,quality=quality,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,candidates_before_source_tx_labels=True,
 caveat='Each record uses received IQ and DURA only: its 20-sample periodic component supplies a pseudo-pilot, not a known true pilot. Clean source IQ is used only for simulation and later quality evaluation. Source IDs overlap prior cycle diagnostics and frozen experts saw source data. Simulated satellite is not validated real satellite; earlier Phase1 test results motivated exploration.'),ensure_ascii=False),flush=True)
'''
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=1800)
'''.replace('INNER', repr(inner), 1)
wrapper = '''import json,subprocess
from pathlib import Path
remote=REMOTE
proc=subprocess.Popen(['C:/Program Files/Git/usr/bin/ssh.exe','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607','python3','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',bufsize=1)
proc.stdin.write(remote);proc.stdin.close();marker='SOURCE_CYCLE_CHANNEL ';payload=None
for line in proc.stdout:
 if line.startswith(marker):payload=line[len(marker):]
 else:print(line,end='',flush=True)
rc=proc.wait()
if rc==0:
 if payload is None:raise ValueError('source result marker missing')
 dest=Path(__file__).parent/'phase1_source_cycle_channel_probe.json'
 if dest.exists():assert json.loads(dest.read_text(encoding='utf8'))==json.loads(payload)
 else:dest.write_text(payload,encoding='utf8',newline='\\n')
 print('SAVED_SOURCE_CYCLE_CHANNEL',dest,flush=True)
raise SystemExit(rc)
'''.replace('REMOTE', repr(remote), 1)
dest = p / 'phase1_source_cycle_channel_probe.py'
assert not dest.exists()
compile(inner, '<source-cycle-channel>', 'exec')
compile(wrapper, str(dest), 'exec')
dest.write_text(wrapper, encoding='utf8', newline='\n')
print('CREATED source pseudo-pilot channel probe', dest)
