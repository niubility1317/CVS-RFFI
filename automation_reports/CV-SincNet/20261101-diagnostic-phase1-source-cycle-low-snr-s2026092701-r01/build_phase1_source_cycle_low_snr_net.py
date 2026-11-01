"""Build source low-SNR weighted, frozen-base distilled cycle denoiser."""
import ast
from pathlib import Path

p = Path(__file__).parent
wrapper = ast.parse((p / 'phase1_source_cycle_feature_net.py').read_text(encoding='utf8'))
node = next(n.value for n in wrapper.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets))
remote_old = ast.literal_eval(node)
rtree = ast.parse(remote_old)
run = next(n for n in ast.walk(rtree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'run')
inner = ast.literal_eval(run.args[0].elts[3])
changes = [
    ("for view in views:\n if view=='clean':received=clean_iq", "for view in views:\n snrs=[]\n if view=='clean':received=clean_iq;snrs=[100.]*n"),
    ("assert np.isfinite(result).all() and not meta['channel_equalization_applied'];received.append(np.stack((result.real,result.imag)).astype(np.float32))", "assert np.isfinite(result).all() and not meta['channel_equalization_applied'];received.append(np.stack((result.real,result.imag)).astype(np.float32));snrs.append(float(meta['signal_to_added_noise_db']))"),
    ("clean_wave=np.stack((clean_aligned.real,clean_aligned.imag),1).astype(np.float32);target_features=[]", "clean_wave=np.stack((clean_aligned.real,clean_aligned.imag),1).astype(np.float32);target_features=[];target_probabilities=[]"),
    ("for start in range(0,n,64):target_features.append(F.normalize(teacher.features(core.tensor(clean_wave[start:start+64],'cpu')),dim=1).cpu())", "for start in range(0,n,64):\n   f=teacher.features(core.tensor(clean_wave[start:start+64],'cpu'));target_features.append(F.normalize(f,dim=1).cpu());target_probabilities.append(F.softmax(teacher.classify_features(f),1).cpu())"),
    ("target_feature=torch.cat(target_features).contiguous(),basep=basep", "target_feature=torch.cat(target_features).contiguous(),target_prob=torch.cat(target_probabilities).contiguous(),snr=core.tensor(np.asarray(snrs,dtype=np.float32),'cpu'),basep=basep"),
    ("torch.nn.Conv2d(6,16,3,padding=1),torch.nn.GELU(),torch.nn.Conv2d(16,16,3,padding=1),torch.nn.GELU(),torch.nn.Conv2d(16,2,3,padding=1)", "torch.nn.Conv2d(6,24,3,padding=1),torch.nn.GELU(),torch.nn.Conv2d(24,24,3,padding=2,dilation=2),torch.nn.GELU(),torch.nn.Conv2d(24,24,3,padding=4,dilation=4),torch.nn.GELU(),torch.nn.Conv2d(24,2,3,padding=1)"),
    ("teacher.eval();teacher.requires_grad_(False);train_anchor=", "teacher.eval();teacher.requires_grad_(False);train_snr=torch.cat([data[v]['snr'] for v in views]);train_prob=torch.cat([data[v]['target_prob'] for v in views]);train_anchor="),
    ("feat=F.normalize(teacher.features(wave),dim=1);feature_loss=(1-(feat*train_feature[idx]).sum(1)).mean();loss=(pred-train_target[idx]).square().mean()+.05*pred.square().mean()+.05*feature_loss", "raw_feat=teacher.features(wave);feat=F.normalize(raw_feat,dim=1);feature_loss=1-(feat*train_feature[idx]).sum(1);logp=F.log_softmax(teacher.classify_features(raw_feat),1);pclean=train_prob[idx];distill=(pclean*(pclean.clamp_min(1e-8).log()-logp)).sum(1);per=(pred-train_target[idx]).square().mean((1,2))+.05*pred.square().mean((1,2))+.05*feature_loss+.02*distill;weight=torch.where(train_snr[idx]<10.,4.,1.);loss=(per*weight).sum()/weight.sum()"),
]
for old, new in changes:
    assert inner.count(old) == 1, old[:100]
    inner = inner.replace(old, new)
inner = inner.replace('SOURCE_CYCLE_FEATURE_NET ', 'SOURCE_CYCLE_LOW_SNR_NET ')
inner = inner.replace('SOURCE_ONLY_CYCLE_FEATURE_NET', 'SOURCE_ONLY_CYCLE_LOW_SNR_NET')
inner = inner.replace("kind='cycle_feature_net'", "kind='cycle_low_snr_net'")
inner = inner.replace("training_records=int(len(train)),training_loss=training_loss", "training_records=int(len(train)),training_low_snr_records=int((np.from_dlpack(train_snr.contiguous())[train]<10.).sum()),training_loss=training_loss")
inner = inner.replace('The regressor uses source clean IQ and frozen Tri-SIG-TIME SSL features aligned to DURA as training targets', 'The regressor uses source clean IQ and frozen Tri-SIG-TIME SSL features and class probabilities aligned to DURA as training targets; simulator SNR weights source training only and is never an inference input')
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=3600)
'''.replace('INNER', repr(inner), 1)
wrapper_new = '''import json,subprocess
from pathlib import Path
remote=REMOTE
proc=subprocess.Popen(['C:/Program Files/Git/usr/bin/ssh.exe','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607','python3','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',bufsize=1)
proc.stdin.write(remote);proc.stdin.close();marker='SOURCE_CYCLE_LOW_SNR_NET ';payload=None
for line in proc.stdout:
 if line.startswith(marker):payload=line[len(marker):]
 else:print(line,end='',flush=True)
rc=proc.wait()
if rc==0:
 if payload is None:raise ValueError('source result marker missing')
 dest=Path(__file__).parent/'phase1_source_cycle_low_snr_net.json'
 if dest.exists():assert json.loads(dest.read_text(encoding='utf8'))==json.loads(payload)
 else:dest.write_text(payload,encoding='utf8',newline='\\n')
 print('SAVED_SOURCE_CYCLE_LOW_SNR_NET',dest,flush=True)
raise SystemExit(rc)
'''.replace('REMOTE', repr(remote), 1)
dest = p / 'phase1_source_cycle_low_snr_net.py'
assert not dest.exists()
compile(inner, '<cycle-low-snr-inner>', 'exec')
compile(wrapper_new, str(dest), 'exec')
dest.write_text(wrapper_new, encoding='utf8', newline='\n')
print('CREATED source cycle low-SNR network', dest)
