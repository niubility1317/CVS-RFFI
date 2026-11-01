"""Replay a fixed cycle network with clean paired targets removed from inference."""
import ast
import json
import subprocess
from pathlib import Path

p = Path(__file__).parent
result = json.loads((p / 'phase1_source_cycle_low_snr_net.json').read_text(encoding='utf8'))
assert result['selected']['kind'] == 'cycle_low_snr_net' and result['heldout'] is not None
wrapper = ast.parse((p / 'phase1_source_cycle_low_snr_net.py').read_text(encoding='utf8'))
remote_node = next(n.value for n in wrapper.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets))
remote_original = ast.literal_eval(remote_node)
rtree = ast.parse(remote_original)
run = next(n for n in ast.walk(rtree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'run')
inner = ast.literal_eval(run.args[0].elts[3])
old = 'n=len(fixed);assert n>300'
new = old + ";sel=np.flatnonzero(split==2);fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel];n=len(fixed);assert n>200"
assert inner.count(old) == 1
inner = inner.replace(old, new)
old = 'clean_aligned=align_to_anchor(clean_complex,z)'
assert inner.count(old) == 1
inner = inner.replace(old, 'clean_aligned=z.copy()')
old_snr = "snrs.append(float(meta['signal_to_added_noise_db']))"
assert inner.count(old_snr) == 1
inner = inner.replace(old_snr, 'snrs.append(0.)')
marker = 'teacher.eval();teacher.requires_grad_(False);train_snr='
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + r'''
import sys
payload=json.load(sys.stdin);model=CycleNet();model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in payload['model_state'].items()});model.eval();predictions={};waves={}
with torch.no_grad():
 for view in views:
  v=data[view];parts=[]
  for start in range(0,n,128):parts.append(model(v['input'][start:start+128]).cpu())
  d=np.from_dlpack(torch.cat(parts).contiguous());delta=(d[:,0].astype(np.float64)+1j*d[:,1])*v['rms'][:,None]/v['rotation'];wave=v['anchor']+payload['alpha']*v['gate'][:,None]*delta;waves[view]=wave;wiq=np.stack((wave.real,wave.imag),1).astype(np.float32);out=[]
  for start in range(0,n,64):
   x=core.tensor(wiq[start:start+64],'cpu');out.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu())
  predictions[view]=np.from_dlpack(torch.cat(out).contiguous()).argmax(1)
print('CYCLE_NET_REPLAY_WAVEFORMS_FIXED_WITHOUT_CLEAN_TARGET_OR_SOURCE_TX_LABELS',flush=True)
def aligned_mse(wave,truth):
 pad=np.pad(wave,((0,0),(16,16)),mode='reflect');truth=truth[:,16:-16];best=np.full(len(wave),np.inf)
 for shift in range(-16,17):
  part=pad[:,16+shift:16+shift+256][:,16:-16];gain=np.sum(np.conj(part)*truth,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-truth)**2,axis=1);best=np.minimum(best,mse)
 return best
quality={}
for view in views:
 before=aligned_mse(data[view]['anchor'],clean_complex);after=aligned_mse(waves[view],clean_complex);quality[view]=dict(count=n,aligned_mse_before=float(before.mean()),aligned_mse_after=float(after.mean()),improved=int((after<before).sum()),worsened=int((after>before).sum()))
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=data['clean']['rawp'];teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target);per={}
for view in views:
 base=data[view]['basep'].argmax(1);out=predictions[view];old=base==target;new=out==target;lm=valid&labeled
 per[view]=dict(valid_count=int(valid.sum()),L_s_count=int(lm.sum()),U_confident_count=int((valid&~labeled).sum()),baseline=int((valid&old).sum()),student=int((valid&new).sum()),repairs=int((valid&~old&new).sum()),damages=int((valid&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()))
base=data['clean']['basep'].argmax(1);clean_net=int((predictions['clean']==teacher_target).sum()-(base==teacher_target).sum())
print('SOURCE_CYCLE_NET_REPLAY '+json.dumps(dict(source_ids=n,heldout=per,heldout_clean_teacher_net=clean_net,quality=quality,paired_clean_targets_used_for_inference=False,simulator_snr_used_for_inference=False,predictions_before_source_tx_labels=True,test_IQ_reads=0),ensure_ascii=False),flush=True)
'''
payload = dict(model_state=result['model_state'], alpha=result['selected']['alpha'])
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,input=PAYLOAD,text=True,check=True,timeout=1800)
'''.replace('INNER', repr(inner), 1).replace('PAYLOAD', repr(json.dumps(payload, separators=(',', ':'))), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=1820)
stdout = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(s for s in stdout.splitlines() if not s.startswith('SOURCE_CYCLE_NET_REPLAY ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
line = next(s[len('SOURCE_CYCLE_NET_REPLAY '):] for s in stdout.splitlines() if s.startswith('SOURCE_CYCLE_NET_REPLAY '))
replay = json.loads(line)
assert replay['source_ids'] == result['heldout_ids'] and replay['heldout'] == result['heldout']
assert replay['heldout_clean_teacher_net'] == result['heldout_clean_teacher_net'] and replay['predictions_before_source_tx_labels']
assert not replay['paired_clean_targets_used_for_inference'] and not replay['simulator_snr_used_for_inference'] and replay['test_IQ_reads'] == 0
for view, q in replay['quality'].items():
    expected = result['quality'][view]
    assert q['count'] == expected['count'] and abs(q['aligned_mse_before'] - expected['aligned_mse_before']) < 1e-8 and abs(q['aligned_mse_after'] - expected['aligned_mse_after']) < 1e-8
dest = p / 'phase1_source_cycle_low_snr_net_replay.json'
if dest.exists():
    assert json.loads(dest.read_text(encoding='utf8')) == replay
else:
    dest.write_text(json.dumps(replay, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print('VERIFIED fixed low-SNR network without clean or simulator-SNR inference inputs', dest)
