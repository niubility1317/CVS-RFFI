"""Build fixed low-SNR network + blind noise gate on disjoint source IDs."""
import ast
from pathlib import Path

p = Path(__file__).parent
wrapper = ast.parse((p / 'phase1_source_cycle_low_snr_net.py').read_text(encoding='utf8'))
node = next(n.value for n in wrapper.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets))
remote_old = ast.literal_eval(node)
rtree = ast.parse(remote_old)
run = next(n for n in ast.walk(rtree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'run')
inner = ast.literal_eval(run.args[0].elts[3])
start = inner.index('q_rng=np.random.default_rng(2026110251)')
end = inner.index('fixed=np.concatenate(groups)', start)
sampling = r'''q_rng=np.random.default_rng(2026110251);h_rng=np.random.default_rng(2026110351);guard_rng=np.random.default_rng(2026110361);utility_rng=np.random.default_rng(2026110371);cycle_rng=np.random.default_rng(2026110411);snr_rng=np.random.default_rng(2026110451);fresh_rng=np.random.default_rng(2026110461);groups=[]
for role,q_count,h_count,guard_count,take,snr_count in (('L_s',2048,512,2048,1024,2048),('U_s',6144,1536,6144,3072,6144)):
 all_ids=np.flatnonzero(roles==role);q=q_rng.choice(all_ids,size=q_count,replace=False);h=h_rng.choice(all_ids,size=h_count,replace=False);without_h=np.setdiff1d(all_ids,np.sort(h),assume_unique=True);guard=guard_rng.choice(without_h,size=guard_count,replace=False);excluded=np.unique(np.concatenate((q,h,guard)));available=np.setdiff1d(all_ids,excluded,assume_unique=True);utility=utility_rng.choice(available,size=take,replace=False);available=np.setdiff1d(available,np.sort(utility),assume_unique=True);cycle=cycle_rng.choice(available,size=take,replace=False);snr=snr_rng.choice(all_ids,size=snr_count,replace=False);available=np.setdiff1d(all_ids,np.unique(np.concatenate((cycle,snr))),assume_unique=True);assert len(available)>=snr_count;fresh=np.sort(fresh_rng.choice(available,size=snr_count,replace=False));assert not np.intersect1d(fresh,cycle).size and not np.intersect1d(fresh,snr).size;groups.append(fresh)
'''
inner = inner[:start] + sampling + inner[end:]
assert inner.count('len(fixed)==len(np.unique(fixed))==4096') == 1
inner = inner.replace('len(fixed)==len(np.unique(fixed))==4096', 'len(fixed)==len(np.unique(fixed))==8192')
assert inner.count('keep=split>=0') == 1
inner = inner.replace('keep=split>=0', 'keep=split>0')
inner = inner.replace('2026110413', '2026110463').replace('2026110412', '2026110462').replace('source-cycle-projection-2026110462', 'source-gated-low-snr-2026110462')
assert inner.count('clean_aligned=align_to_anchor(clean_complex,z)') == 1
inner = inner.replace('clean_aligned=align_to_anchor(clean_complex,z)', 'clean_aligned=z.copy()')
old = 'for view in views:\n snrs=[]'
helper = r'''def wide_mse(wave,truth):
 pad=np.pad(wave,((0,0),(64,64)),mode='reflect');target=truth[:,64:-64];best=np.full(len(wave),np.inf)
 for shift in range(-64,65):
  part=pad[:,64+shift:64+shift+256][:,64:-64];gain=np.sum(np.conj(part)*target,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-target)**2,axis=1);best=np.minimum(best,mse)
 return best
'''
assert inner.count(old) == 1
inner = inner.replace(old, helper + old)
old = "data[view]=dict(input=core.tensor(features,'cpu'),target=core.tensor(target_iq,'cpu'),anchor=z,"
new = "raw_energy=np.mean(abs(received_complex[:,64:-64])**2,axis=1).clip(min=1e-20);fraction=np.clip(wide_mse(z,received_complex)/raw_energy,1e-8,1.);blind_snr=10*np.log10(np.maximum(1-fraction,1e-8)/fraction);data[view]=dict(blind_snr=blind_snr,input=core.tensor(features,'cpu'),target=core.tensor(target_iq,'cpu'),anchor=z,"
assert inner.count(old) == 1
inner = inner.replace(old, new)
marker = 'teacher.eval();teacher.requires_grad_(False);train_snr='
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + r'''
import sys
payload=json.load(sys.stdin);assert payload['alpha']==.25 and payload['gate_db']==10.;model=CycleNet();model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in payload['model_state'].items()});model.eval();predictions={};waves={}
with torch.no_grad():
 for view in views:
  v=data[view];parts=[]
  for start in range(0,n,128):parts.append(model(v['input'][start:start+128]).cpu())
  d=np.from_dlpack(torch.cat(parts).contiguous());delta=(d[:,0].astype(np.float64)+1j*d[:,1])*v['rms'][:,None]/v['rotation'];ungated=v['anchor']+payload['alpha']*v['gate'][:,None]*delta;gate=v['gate']&(v['blind_snr']<payload['gate_db']);gated=v['anchor']+payload['alpha']*gate[:,None]*delta;waves[view]=dict(ungated=ungated,gated=gated);predictions[view]={}
  for method,wave in waves[view].items():
   wiq=np.stack((wave.real,wave.imag),1).astype(np.float32);out=[]
   for start in range(0,n,64):
    x=core.tensor(wiq[start:start+64],'cpu');out.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu())
   predictions[view][method]=np.from_dlpack(torch.cat(out).contiguous()).argmax(1)
print('FRESH_GATED_LOW_SNR_WAVEFORMS_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=data['clean']['rawp'];teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target)
def aligned_mse(wave,truth):
 pad=np.pad(wave,((0,0),(16,16)),mode='reflect');truth=truth[:,16:-16];best=np.full(len(wave),np.inf)
 for shift in range(-16,17):
  part=pad[:,16+shift:16+shift+256][:,16:-16];gain=np.sum(np.conj(part)*truth,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-truth)**2,axis=1);best=np.minimum(best,mse)
 return best
def evaluate(part,group=None):
 mask=split==part
 if group is not None:mask &= source_rx==group
 vm=mask&valid;lm=vm&labeled;results={}
 for view in views:
  v=data[view];base=v['basep'].argmax(1);raw_correct=base==target;row=dict(count=int(mask.sum()),valid_count=int(vm.sum()),L_s_count=int(lm.sum()),U_confident_count=int((vm&~labeled).sum()),baseline=int((vm&raw_correct).sum()),L_s_baseline=int((lm&raw_correct).sum()),gate_enabled=int((mask&v['gate']&(v['blind_snr']<payload['gate_db'])).sum()))
  for method in ('ungated','gated'):
   out=predictions[view][method];ok=out==target;row[method]=dict(correct=int((vm&ok).sum()),repairs=int((vm&~raw_correct&ok).sum()),damages=int((vm&raw_correct&~ok).sum()),L_s_correct=int((lm&ok).sum()))
  snr=np.from_dlpack(v['snr'].contiguous());truth=snr<10.;selected=v['blind_snr']<payload['gate_db'];row['blind_snr_confusion']=dict(true_low=int((mask&truth).sum()),true_positive=int((mask&truth&selected).sum()),false_positive=int((mask&~truth&selected).sum()),false_negative=int((mask&truth&~selected).sum()),true_negative=int((mask&~truth&~selected).sum()));results[view]=row
 return results
quality={}
for view in views:
 v=data[view];m=split==2;truth=clean_complex[m];base=aligned_mse(v['anchor'][m],truth);q=dict(count=int(m.sum()),baseline=float(base.mean()))
 for method in ('ungated','gated'):
  err=aligned_mse(waves[view][method][m],truth);q[method]=dict(aligned_mse=float(err.mean()),improved=int((err<base).sum()),worsened=int((err>base).sum()))
 quality[view]=q
print('SOURCE_GATED_LOW_SNR '+json.dumps(dict(status='SOURCE_ONLY_FRESH_GATED_LOW_SNR',source_ids=n,validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),gate_db=payload['gate_db'],alpha=payload['alpha'],validation=evaluate(1),validation_groups={str(g):evaluate(1,g) for g in (4,6)},heldout=evaluate(2),quality=quality,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,predictions_before_source_tx_labels=True,clean_target_used_for_inference=False,simulated_snr_used_for_inference=False,
 caveat='This fixed model and gate were chosen after earlier source results. Fresh IDs exclude the complete 4096-record cycle-network sample and 8192-record SNR/proxy sample, but may overlap other source diagnostics. New channel and hardware seeds are used. Only source simulator uses source clean IQ and simulator SNR; inference uses received IQ, DURA and the fixed network, with blind SNR computed from received/DURA. Receiver/day holdout is disjoint for this network, not for frozen experts. Simulated satellite is not a validated real link.'),ensure_ascii=False),flush=True)
'''
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,input=PAYLOAD,text=True,check=True,timeout=1800)
'''
wrapper_new = '''import json,subprocess
from pathlib import Path
p=Path(__file__).parent
source=json.loads((p/'phase1_source_cycle_low_snr_net.json').read_text(encoding='utf8'))
assert source['selected']=={'kind':'cycle_low_snr_net','epoch':2,'alpha':.25,'validation_net':1}
payload={'model_state':source['model_state'],'alpha':.25,'gate_db':10.}
inner=INNER
remote=REMOTE.replace('INNER',repr(inner),1).replace('PAYLOAD',repr(json.dumps(payload,separators=(',',':'))),1)
proc=subprocess.Popen(['C:/Program Files/Git/usr/bin/ssh.exe','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607','python3','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',bufsize=1)
proc.stdin.write(remote);proc.stdin.close();marker='SOURCE_GATED_LOW_SNR ';payload_line=None
for line in proc.stdout:
 if line.startswith(marker):payload_line=line[len(marker):]
 else:print(line,end='',flush=True)
rc=proc.wait()
if rc==0:
 if payload_line is None:raise ValueError('fresh source result marker missing')
 result=json.loads(payload_line)
 assert result['U_s_labels_read']==result['V_reads']==result['test_IQ_reads']==0 and result['predictions_before_source_tx_labels']
 assert not result['clean_target_used_for_inference'] and not result['simulated_snr_used_for_inference']
 dest=p/'phase1_source_gated_low_snr_probe.json'
 if dest.exists():assert json.loads(dest.read_text(encoding='utf8'))==result
 else:dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\\n',encoding='utf8',newline='\\n')
 print('SAVED_SOURCE_GATED_LOW_SNR',dest,flush=True)
raise SystemExit(rc)
'''.replace('INNER', repr(inner), 1).replace('REMOTE', repr(remote), 1)
dest = p / 'phase1_source_gated_low_snr_probe.py'
if dest.exists():
    assert 'SOURCE_GATED_LOW_SNR' in dest.read_text(encoding='utf8')
compile(inner, '<fresh-gated-low-snr>', 'exec')
compile(wrapper_new, str(dest), 'exec')
dest.write_text(wrapper_new, encoding='utf8', newline='\n')
print('CREATED fixed blind-gated fresh-source probe', dest)
