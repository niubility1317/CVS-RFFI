"""Add a frozen-base source feature constraint to the cycle residual network."""
import ast
from pathlib import Path

p = Path(__file__).parent
wrapper = ast.parse((p / 'phase1_source_cycle_residual_net.py').read_text(encoding='utf8'))
node = next(n.value for n in wrapper.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets))
remote_old = ast.literal_eval(node)
rtree = ast.parse(remote_old)
run = next(n for n in ast.walk(rtree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'run')
inner = ast.literal_eval(run.args[0].elts[3])
old = " data[view]=dict(input=core.tensor(features,'cpu'),target=core.tensor(target_iq,'cpu'),anchor=z,basep=basep,rawp=rawp,rotation=rotation,rms=rms,gate=gate);print('CYCLE_NET_SOURCE_INPUTS_FIXED',view,flush=True)"
new = r''' clean_wave=np.stack((clean_aligned.real,clean_aligned.imag),1).astype(np.float32);target_features=[]
 with torch.no_grad():
  for start in range(0,n,64):target_features.append(F.normalize(teacher.features(core.tensor(clean_wave[start:start+64],'cpu')),dim=1).cpu())
 rotation_inv=1/rotation;rotation_iq=np.stack((rotation_inv.real,rotation_inv.imag),1).astype(np.float32);anchor_iq=np.stack((z.real,z.imag),1).astype(np.float32)
 data[view]=dict(input=core.tensor(features,'cpu'),target=core.tensor(target_iq,'cpu'),anchor=z,anchor_t=core.tensor(anchor_iq,'cpu'),rotation_t=core.tensor(rotation_iq,'cpu'),rms_t=core.tensor(rms.astype(np.float32),'cpu'),target_feature=torch.cat(target_features).contiguous(),basep=basep,rawp=rawp,rotation=rotation,rms=rms,gate=gate);print('CYCLE_FEATURE_SOURCE_INPUTS_FIXED',view,flush=True)'''
assert inner.count(old) == 1
inner = inner.replace(old, new)
old = "train_input=torch.cat([data[v]['input'] for v in views]);train_target=torch.cat([data[v]['target'] for v in views]);train_split="
new = "teacher.eval();teacher.requires_grad_(False);train_anchor=torch.cat([data[v]['anchor_t'] for v in views]);train_rotation=torch.cat([data[v]['rotation_t'] for v in views]);train_rms=torch.cat([data[v]['rms_t'] for v in views]);train_feature=torch.cat([data[v]['target_feature'] for v in views]);train_input=torch.cat([data[v]['input'] for v in views]);train_target=torch.cat([data[v]['target'] for v in views]);train_split="
assert inner.count(old) == 1
inner = inner.replace(old, new)
old = "idx=torch.tensor(order[start:start+64].tolist(),dtype=torch.long);pred=model(train_input[idx]);loss=(pred-train_target[idx]).square().mean()+.05*pred.square().mean()"
new = r'''idx=torch.tensor(order[start:start+64].tolist(),dtype=torch.long);pred=model(train_input[idx]);rot=train_rotation[idx];amp=train_rms[idx,None];anchor=train_anchor[idx];real=anchor[:,0]+amp*(pred[:,0]*rot[:,0]-pred[:,1]*rot[:,1]);imag=anchor[:,1]+amp*(pred[:,0]*rot[:,1]+pred[:,1]*rot[:,0]);wave=torch.stack((real,imag),1);feat=F.normalize(teacher.features(wave),dim=1);feature_loss=(1-(feat*train_feature[idx]).sum(1)).mean();loss=(pred-train_target[idx]).square().mean()+.05*pred.square().mean()+.05*feature_loss'''
assert inner.count(old) == 1
inner = inner.replace(old, new)
inner = inner.replace('SOURCE_CYCLE_NET ', 'SOURCE_CYCLE_FEATURE_NET ')
inner = inner.replace('SOURCE_ONLY_CYCLE_RESIDUAL_NET', 'SOURCE_ONLY_CYCLE_FEATURE_NET')
inner = inner.replace("kind='cycle_residual_net'", "kind='cycle_feature_net'")
inner = inner.replace('The regressor uses source clean IQ aligned to DURA as a training target', 'The regressor uses source clean IQ and frozen Tri-SIG-TIME SSL features aligned to DURA as training targets')
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
proc.stdin.write(remote);proc.stdin.close();marker='SOURCE_CYCLE_FEATURE_NET ';payload=None
for line in proc.stdout:
 if line.startswith(marker):payload=line[len(marker):]
 else:print(line,end='',flush=True)
rc=proc.wait()
if rc==0:
 if payload is None:raise ValueError('source result marker missing')
 dest=Path(__file__).parent/'phase1_source_cycle_feature_net.json'
 if dest.exists():assert json.loads(dest.read_text(encoding='utf8'))==json.loads(payload)
 else:dest.write_text(payload,encoding='utf8',newline='\\n')
 print('SAVED_SOURCE_CYCLE_FEATURE_NET',dest,flush=True)
raise SystemExit(rc)
'''.replace('REMOTE', repr(remote), 1)
dest = p / 'phase1_source_cycle_feature_net.py'
assert not dest.exists()
compile(inner, '<cycle-feature-inner>', 'exec')
compile(wrapper_new, str(dest), 'exec')
dest.write_text(wrapper_new, encoding='utf8', newline='\n')
print('CREATED source cycle feature network', dest)
