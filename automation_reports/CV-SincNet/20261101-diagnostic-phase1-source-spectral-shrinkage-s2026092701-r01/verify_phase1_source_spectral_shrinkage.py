"""Replay the fixed source spectral network without paired-clean inputs."""
import ast,json,subprocess
from pathlib import Path
p=Path(__file__).parent
result=json.loads((p/'phase1_source_spectral_shrinkage_network.json').read_text(encoding='utf8'))
assert result['selected']['kind']=='spectral_shrinkage_network' and result['model_state']
tree=ast.parse((p/'phase1_source_cycle_lowrank_probe.py').read_text(encoding='utf8'))
inner=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='inner' for t in n.targets)))
inner=inner.replace('2026110521','2026110531').replace('2026110522','2026110532').replace('2026110523','2026110533')
inner=inner.replace('keep=split>0','split[np.isin(source_rx,[1,3,4])&np.isin(source_day,[1,2])]=0;keep=split>=0')
inner=inner.split("for view in views:\n if view=='clean':received=clean_iq",1)[0]
tree=ast.parse((p/'phase1_source_spectral_shrinkage_network.py').read_text(encoding='utf8'))
extra=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='inner'))
extra=extra.split('torch.manual_seed(2026110534);model=SpectralGate();',1)[0]
a=extra.index('teacher.eval();teacher.requires_grad_(False);ground_features=[];ground_probs=[]');b=extra.index('for view in views:',a)
extra=extra[:a]+'teacher.eval();teacher.requires_grad_(False)\n'+extra[b:]
assert extra.count('target=align_clean_to_anchor(clean_complex,z)')==1
extra=extra.replace('target=align_clean_to_anchor(clean_complex,z)','target=np.zeros_like(z)')
inner+=extra
inner+='\nFIXED_STATE='+repr(result['model_state'])+'\nFIXED_ALPHA='+repr(result['selected']['alpha'])+'\n'
inner+=r'''
model=SpectralGate();model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in FIXED_STATE.items()});model.eval();eval_ids=np.flatnonzero(split==2);predictions={};waves={}
with torch.no_grad():
 for view in views:
  d=data[view];prob=[];out=[]
  for start in range(0,len(eval_ids),64):
   ix=torch.tensor(eval_ids[start:start+64].tolist(),dtype=torch.long);g=model(d['features'][ix]);delta=torch.einsum('bk,bkct->bct',g,d['basis'][ix]);gate=torch.tensor(d['gate'][eval_ids[start:start+64]].astype(np.float32).tolist())[:,None,None];wave=d['anchor'][ix]-FIXED_ALPHA*gate*delta;out.append(wave.cpu());prob.append(F.softmax(teacher.classify_features(teacher.features(wave)),1).cpu())
  predictions[view]=np.from_dlpack(torch.cat(prob).contiguous());waves[view]=np.from_dlpack(torch.cat(out).contiguous());print('FIXED_SPECTRAL_REPLAY_INFERENCE_COMPLETE',view,flush=True)
print('ALL_FIXED_SPECTRAL_REPLAY_PREDICTIONS_BEFORE_SOURCE_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=data['clean']['rawp'];teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target);mask=valid[eval_ids];heldout={};quality={}
for view in views:
 d=data[view];base=d['basep'][eval_ids].argmax(1);pred=predictions[view].argmax(1);truth=target[eval_ids];old=base==truth;new=pred==truth;lm=mask&labeled[eval_ids]
 heldout[view]=dict(valid_count=int(mask.sum()),L_s_count=int(lm.sum()),baseline=int((mask&old).sum()),student=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()))
 base_iq=np.from_dlpack(d['anchor'][torch.tensor(eval_ids.tolist(),dtype=torch.long)].contiguous());wave=waves[view];b=base_iq[:,0].astype(np.float64)+1j*base_iq[:,1];w=wave[:,0].astype(np.float64)+1j*wave[:,1];truth_iq=clean_complex[eval_ids];before=wide_mse(b,truth_iq);after=wide_mse(w,truth_iq);quality[view]=dict(count=int(len(eval_ids)),aligned_mse_before=float(before.mean()),aligned_mse_after=float(after.mean()),improved=int((after<before).sum()),worsened=int((after>before).sum()))
print('SOURCE_SPECTRAL_FIXED_REPLAY '+json.dumps(dict(status='SOURCE_FIXED_SPECTRAL_REPLAY_WITH_DUMMY_CLEAN_TARGET',heldout=heldout,quality=quality,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,candidate_inference_uses_clean_target=False,candidate_inference_uses_simulator_snr=False),ensure_ascii=False),flush=True)
'''
compile(inner,'<fixed-spectral-replay>','exec')
remote='''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=1800)
'''.replace('INNER',repr(inner),1)
proc=subprocess.Popen(['C:/Program Files/Git/usr/bin/ssh.exe','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607','python3','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',bufsize=1)
proc.stdin.write(remote);proc.stdin.close();payload=None
for line in proc.stdout:
 if line.startswith('SOURCE_SPECTRAL_FIXED_REPLAY '):payload=line[len('SOURCE_SPECTRAL_FIXED_REPLAY '):]
 else:print(line,end='',flush=True)
rc=proc.wait()
if rc:raise SystemExit(rc)
if payload is None:raise ValueError('missing fixed replay')
replay=json.loads(payload);assert replay['heldout']==result['heldout'];assert replay['quality']==result['quality'];assert replay['U_s_labels_read']==replay['V_reads']==replay['test_IQ_reads']==0
dest=p/'phase1_source_spectral_shrinkage_replay.json';dest.write_text(json.dumps(replay,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n');print('VERIFIED_FIXED_SPECTRAL_REPLAY',dest,flush=True)
