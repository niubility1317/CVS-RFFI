"""Replay source-holdout carrier model before joining transmitter labels."""
import ast
import json
import subprocess
from pathlib import Path

p = Path(__file__).parent
result = json.loads((p / 'phase1_source_carrier_regression_tolerance.json').read_text(encoding='utf8'))
assert result['selected']['kind'] == 'carrier_phase_regressor' and result['heldout'] is not None
tree = ast.parse((p / 'phase1_source_carrier_regression_tolerance.py').read_text(encoding='utf8'))
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
old = "assert all(int((split==i).sum())>200 for i in range(3));n=len(fixed);views="
new = "assert all(int((split==i).sum())>200 for i in range(3));sel=np.flatnonzero(split==2);fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel];n=len(fixed);views="
assert inner.count(old) == 1
inner = inner.replace(old, new)
marker = "all_input=torch.cat([data[v]['input'] for v in views])"
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + r'''
import sys
payload=json.load(sys.stdin);model=PhaseNet();model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in payload['model_state'].items()});model.eval();outputs={};quality={}
with torch.no_grad():
 for view in views:
  v=data[view];raw=[]
  for start in range(0,n,256):raw.append(model(v['input'][start:start+256]).cpu())
  est=torch.cat(raw);theta=torch.atan2(est[:,1],est[:,0]);conf=est.square().sum(1).sqrt();apply=(conf>=payload['threshold'])&(v['ratio']>.2);theta=torch.where(apply,theta,torch.zeros_like(theta));pred=[]
  for start in range(0,n,64):
   a=v['anchor'][start:start+64];z=torch.complex(a[:,0],a[:,1])*torch.exp(1j*theta[start:start+64,None]);wave=torch.stack((z.real,z.imag),1);pred.append(F.softmax(teacher.classify_features(teacher.features(wave)),1).cpu())
  outputs[view]=np.from_dlpack(torch.cat(pred).contiguous()).argmax(1);target_angle=torch.atan2(v['target'][:,1],v['target'][:,0]);w=v['weight'];quality[view]=dict(correlation_weighted_before=float(((1-torch.cos(target_angle))*w).sum()/w.sum().clamp_min(1e-8)),correlation_weighted_after=float(((1-torch.cos(theta-target_angle))*w).sum()/w.sum().clamp_min(1e-8)),applied=int(apply.sum()))
print('CARRIER_REPLAY_WAVEFORMS_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=np.from_dlpack(data['clean']['rawp']);teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target);per={}
for view in views:
 base=np.from_dlpack(data[view]['basep']).argmax(1);out=outputs[view];old=base==target;new=out==target;lm=valid&labeled
 per[view]=dict(valid_count=int(valid.sum()),L_s_count=int(lm.sum()),U_confident_count=int((valid&~labeled).sum()),baseline=int((valid&old).sum()),student=int((valid&new).sum()),repairs=int((valid&~old&new).sum()),damages=int((valid&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()))
clean_base=np.from_dlpack(data['clean']['basep']).argmax(1);clean_net=int((outputs['clean']==teacher_target).sum()-(clean_base==teacher_target).sum())
print('SOURCE_CARRIER_REPLAY '+json.dumps(dict(heldout=per,heldout_clean_teacher_net=clean_net,source_ids=n,quality=quality,test_IQ_reads=0,predictions_before_source_tx_labels=True),ensure_ascii=False),flush=True)
'''
payload = dict(model_state=result['model_state'], threshold=result['selected']['threshold'])
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,input=PAYLOAD,text=True,check=True,timeout=1800)
'''.replace('INNER', repr(inner), 1).replace('PAYLOAD', repr(json.dumps(payload, separators=(',', ':'))), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=1820)
stdout = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(line for line in stdout.splitlines() if not line.startswith('SOURCE_CARRIER_REPLAY ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
line = next(x[len('SOURCE_CARRIER_REPLAY '):] for x in stdout.splitlines() if x.startswith('SOURCE_CARRIER_REPLAY '))
replay = json.loads(line)
assert replay['heldout'] == result['heldout'] and replay['heldout_clean_teacher_net'] == result['heldout_clean_teacher_net']
assert replay['source_ids'] == result['heldout_ids'] and replay['predictions_before_source_tx_labels'] and replay['test_IQ_reads'] == 0
dest = p / 'phase1_source_carrier_regression_replay.json'
if dest.exists():
    assert json.loads(dest.read_text(encoding='utf8')) == replay
else:
    dest.write_text(json.dumps(replay, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print('VERIFIED independent source carrier replay', dest)
