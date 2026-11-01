"""Replay the fixed source spectral network without paired-clean inputs."""
import ast,json,subprocess
from pathlib import Path
p=Path(__file__).parent
result=json.loads((p/'phase1_source_spectral_shrinkage_network.json').read_text(encoding='utf8'))
assert result['selected']['kind']=='spectral_shrinkage_network' and result['model_state']
tree=ast.parse((p/'phase1_source_cycle_lowrank_probe.py').read_text(encoding='utf8'))
inner=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='inner' for t in n.targets)))
inner=inner.replace('2026110521','2026110531').replace('2026110522','2026110532').replace('2026110523','2026110533')
a=inner.index('q_rng=np.random.default_rng(2026110251)');b=inner.index('fixed=np.concatenate(groups)',a)
inner=inner[:a]+"q_rng=np.random.default_rng(2026110251);h_rng=np.random.default_rng(2026110351);guard_rng=np.random.default_rng(2026110361);utility_rng=np.random.default_rng(2026110371);old_rng=np.random.default_rng(2026110531);low_rng=np.random.default_rng(2026110521);new_rng=np.random.default_rng(2026110541);groups=[]\nfor role,q_count,h_count,guard_count,take in (('L_s',2048,512,2048,1024),('U_s',6144,1536,6144,3072)):\n all_ids=np.flatnonzero(roles==role);q=q_rng.choice(all_ids,size=q_count,replace=False);h=h_rng.choice(all_ids,size=h_count,replace=False);without_h=np.setdiff1d(all_ids,np.sort(h),assume_unique=True);guard=guard_rng.choice(without_h,size=guard_count,replace=False);available=np.setdiff1d(all_ids,np.unique(np.concatenate((q,h,guard))),assume_unique=True);utility=utility_rng.choice(available,size=take,replace=False);available=np.setdiff1d(available,np.sort(utility),assume_unique=True);old=old_rng.choice(available,size=take,replace=False);low=low_rng.choice(available,size=take,replace=False);eligible=np.setdiff1d(all_ids,np.unique(np.concatenate((old,low))),assume_unique=True);print('FRESH_SOURCE_SAMPLING',role,'available',len(available),'eligible',len(eligible),'required',take,flush=True);assert len(eligible)>=take;new=np.sort(new_rng.choice(eligible,size=take,replace=False));assert not np.intersect1d(ids[new],ids[old]).size and not np.intersect1d(ids[new],ids[low]).size;groups.append(new)\n"+inner[b:]
inner=inner.replace('2026110532','2026110542').replace('2026110533','2026110543')
if '--smoke' in __import__('sys').argv:
 old='n=len(fixed);assert n>300';new=old+";sel=np.concatenate((np.flatnonzero(source_rx==4)[:2],np.flatnonzero(source_rx==6)[:2],np.flatnonzero(source_rx==8)[:2]));fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel];n=len(fixed)";assert inner.count(old)==1;inner=inner.replace(old,new)
inner=inner.split("for view in views:\n if view=='clean':received=clean_iq",1)[0]
tree=ast.parse((p/'phase1_source_spectral_shrinkage_network.py').read_text(encoding='utf8'))
extra=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='inner'))
extra=extra.split('torch.manual_seed(2026110534);model=SpectralGate();',1)[0]
a=extra.index('teacher.eval();teacher.requires_grad_(False);ground_features=[];ground_probs=[]');b=extra.index('for view in views:',a)
extra=extra[:a]+'teacher.eval();teacher.requires_grad_(False)\n'+extra[b:]
assert extra.count('target=align_clean_to_anchor(clean_complex,z)')==1
extra=extra.replace('target=align_clean_to_anchor(clean_complex,z)','target=np.zeros_like(z)')
extra=extra.replace('2026110532','2026110542').replace('2026110533','2026110543')
inner+=extra
inner+='\nFIXED_STATE='+repr(result['model_state'])+'\nFIXED_ALPHA='+repr(result['selected']['alpha'])+'\n'
inner+=r'''
model=SpectralGate();model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in FIXED_STATE.items()});model.eval();eval_ids=np.flatnonzero(split>0);predictions={};waves={}
with torch.no_grad():
 for view in views:
  d=data[view];prob=[];out=[]
  for start in range(0,len(eval_ids),64):
   ix=torch.tensor(eval_ids[start:start+64].tolist(),dtype=torch.long);g=model(d['features'][ix]);delta=torch.einsum('bk,bkct->bct',g,d['basis'][ix]);gate=torch.tensor(d['gate'][eval_ids[start:start+64]].astype(np.float32).tolist())[:,None,None];wave=d['anchor'][ix]-FIXED_ALPHA*gate*delta;out.append(wave.cpu());prob.append(F.softmax(teacher.classify_features(teacher.features(wave)),1).cpu())
  predictions[view]=np.from_dlpack(torch.cat(prob).contiguous());waves[view]=np.from_dlpack(torch.cat(out).contiguous());print('FIXED_SPECTRAL_REPLAY_INFERENCE_COMPLETE',view,flush=True)
print('ALL_FIXED_SPECTRAL_REPLAY_PREDICTIONS_BEFORE_SOURCE_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=data['clean']['rawp'];teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target);results={};quality={}
for part in (1,2):
 mask=valid[eval_ids]&(split[eval_ids]==part);iqmask=split[eval_ids]==part;heldout={};qresult={}
 for view in views:
  d=data[view];base=d['basep'][eval_ids].argmax(1);pred=predictions[view].argmax(1);truth=target[eval_ids];old=base==truth;new=pred==truth;lm=mask&labeled[eval_ids]
  heldout[view]=dict(valid_count=int(mask.sum()),L_s_count=int(lm.sum()),baseline=int((mask&old).sum()),student=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()))
  base_iq=np.from_dlpack(d['anchor'][torch.tensor(eval_ids.tolist(),dtype=torch.long)].contiguous());wave=waves[view];b=base_iq[:,0].astype(np.float64)+1j*base_iq[:,1];w=wave[:,0].astype(np.float64)+1j*wave[:,1];truth_iq=clean_complex[eval_ids];before=wide_mse(b[iqmask],truth_iq[iqmask]);after=wide_mse(w[iqmask],truth_iq[iqmask]);qresult[view]=dict(count=int(iqmask.sum()),aligned_mse_before=float(before.mean()),aligned_mse_after=float(after.mean()),improved=int((after<before).sum()),worsened=int((after>before).sum()))
 results[str(part)]=heldout;quality[str(part)]=qresult
print('SOURCE_SPECTRAL_FRESH_EVAL '+json.dumps(dict(status='SOURCE_ONLY_FRESH_FIXED_SPECTRAL_NETWORK',source_ids=n,validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),results=results,quality=quality,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,decisions_before_source_tx_labels=True,candidate_inference_uses_clean_target=False,candidate_inference_uses_simulator_snr=False,caveat='Fresh source IDs exclude both earlier spectral network and fixed low-rank samples, but may overlap other probes. Fixed teacher and DURA saw source records. Simulated satellite is not a validated real link.'),ensure_ascii=False),flush=True)
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
 if line.startswith('SOURCE_SPECTRAL_FRESH_EVAL '):payload=line[len('SOURCE_SPECTRAL_FRESH_EVAL '):]
 else:print(line,end='',flush=True)
rc=proc.wait()
if rc:raise SystemExit(rc)
if payload is None:raise ValueError('missing fixed replay')
replay=json.loads(payload);assert replay['U_s_labels_read']==replay['V_reads']==replay['test_IQ_reads']==0 and replay['decisions_before_source_tx_labels']
if '--smoke' in __import__('sys').argv:print('VERIFIED_FRESH_SPECTRAL_SMOKE',replay['source_ids']);raise SystemExit(0)
dest=p/'phase1_source_spectral_shrinkage_fresh.json'
if dest.exists():assert json.loads(dest.read_text(encoding='utf8'))==replay
else:dest.write_text(json.dumps(replay,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
print('SAVED_FRESH_SPECTRAL_EVALUATION',dest,flush=True)
