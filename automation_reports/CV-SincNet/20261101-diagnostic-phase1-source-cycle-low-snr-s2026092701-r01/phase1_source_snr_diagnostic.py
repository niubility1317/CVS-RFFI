"""Stratify source-only F and DURA outcomes by privileged simulator SNR."""
import json
import subprocess
import sys
from pathlib import Path

inner = r'''
from pathlib import Path
import json,numpy as np,torch
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,physical_rich_router_experiment as g,feature_conditioned_experiment as fs
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2);models=g.load_models(root,'cpu');teacher,contract,dura,*_=models;cache,clean,roles=core.data(root,contract)
from leo_practical.channel import Config,receiver_for_session,stable_seed,normalize_input
from leo_practical.residual import ResidualChannel
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);rx=np.load(cache/'train/rx.npy',allow_pickle=False);day=np.load(cache/'train/day.npy',allow_pickle=False);rng=np.random.default_rng(2026110451);groups=[]
for role,take in (('L_s',2048),('U_s',6144)):
 available=np.flatnonzero(roles==role);groups.append(np.sort(rng.choice(available,size=take,replace=False)))
fixed=np.concatenate(groups);assert len(fixed)==len(np.unique(fixed))==8192;keep=(rx[fixed]==8)&(day[fixed]==3);fixed=fixed[keep];n=len(fixed);assert n>300;views=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban');clean_iq=np.asarray(clean[fixed],dtype=np.float32);clean_complex=clean_iq[:,0].astype(np.float64)+1j*clean_iq[:,1];data={}
def aligned_mse(wave,truth):
 pad=np.pad(wave,((0,0),(16,16)),mode='reflect');truth=truth[:,16:-16];best=np.full(len(wave),np.inf)
 for shift in range(-16,17):
  part=pad[:,16+shift:16+shift+256][:,16:-16];gain=np.sum(np.conj(part)*truth,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-truth)**2,axis=1);best=np.minimum(best,mse)
 return best
for view in views:
 snr=[];lock=[];state=[]
 if view=='clean':received=clean_iq
 else:
  cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');received=[]
  for j in fixed:
   pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110453,session);v=np.asarray(clean[j],dtype=np.float64);signal,_=normalize_input(v[0]+1j*v[1]);result,meta=ResidualChannel(cfg,stable_seed(2026110452,'source-snr-diagnostic-2026110452',view,pid),hw).process(signal);assert np.isfinite(result).all() and not meta['channel_equalization_applied'];received.append(np.stack((result.real,result.imag)).astype(np.float32));snr.append(float(meta['signal_to_added_noise_db']));lock.append(str(meta['receiver_lock_status']));state.append(str(meta['state_start']))
  received=np.stack(received)
 rawp=[];basep=[];anchors=[]
 with torch.no_grad():
  for start in range(0,n,32):
   x=core.tensor(received[start:start+32],'cpu');a=dura(x);rawp.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu());basep.append(F.softmax(teacher.classify_features(teacher.features(a)),1).cpu());anchors.append(a.cpu())
 rawp=np.from_dlpack(torch.cat(rawp).contiguous());basep=np.from_dlpack(torch.cat(basep).contiguous());anchor=np.from_dlpack(torch.cat(anchors).contiguous());z=anchor[:,0].astype(np.float64)+1j*anchor[:,1];data[view]=dict(rawp=rawp,basep=basep,snr=np.asarray(snr),lock=np.asarray(lock),state=np.asarray(state),mse=aligned_mse(z,clean_complex));print('SNR_SOURCE_PREDICTIONS_FIXED',view,flush=True)
print('ALL_SNR_SOURCE_PREDICTIONS_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=data['clean']['rawp'];teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target);edges=(-1e9,0,5,10,15,20,25,30,1e9);results={}
def summarize(mask,view):
 v=data[view];m=mask&valid;lm=m&labeled;raw=v['rawp'].argmax(1);base=v['basep'].argmax(1);return dict(count=int(mask.sum()),valid_count=int(m.sum()),L_s_count=int(lm.sum()),raw_correct=int((m&(raw==target)).sum()),dura_correct=int((m&(base==target)).sum()),L_s_raw_correct=int((lm&(raw==target)).sum()),L_s_dura_correct=int((lm&(base==target)).sum()),mean_aligned_mse=float(v['mse'][mask].mean()) if mask.any() else None)
for view in views[1:]:
 v=data[view];groups=[]
 for lo,hi in zip(edges[:-1],edges[1:]):
  m=(v['snr']>=lo)&(v['snr']<hi);groups.append(dict(lower_db=None if lo==-1e9 else lo,upper_db=None if hi==1e9 else hi,**summarize(m,view)))
 status={s:summarize(v['lock']==s,view) for s in sorted(set(v['lock']))};states={s:summarize(v['state']==s,view) for s in sorted(set(v['state']))};results[view]=dict(overall=summarize(np.ones(n,dtype=bool),view),snr_quantiles_db=np.quantile(v['snr'],[.05,.25,.5,.75,.95]).tolist(),snr_bins=groups,receiver_lock=status,propagation_state=states)
print('SOURCE_SNR_DIAGNOSTIC '+json.dumps(dict(status='SOURCE_ONLY_SNR_DIAGNOSTIC',source_ids=n,receiver=8,day=3,clean_labeled_count=int((valid&labeled).sum()),clean_teacher_correct=int((valid&(teacher_target==target)).sum()),results=results,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,predictions_before_source_tx_labels=True,
 caveat='SNR, receiver lock and propagation state are privileged simulator metadata used only for post-hoc source stratification, not available inference inputs. SNR is simulated added-noise SNR, not total measured SNR or validated real satellite SNR. Source IQ IDs may overlap other diagnostics; frozen models saw source records. Valid targets mix true L_s with confident clean-teacher pseudo-labels.'),ensure_ascii=False),flush=True)
'''
smoke = '--smoke' in sys.argv
if smoke:
    old = 'n=len(fixed);assert n>300'
    new = old + ";sel=np.concatenate((np.flatnonzero(roles[fixed]=='L_s')[:2],np.flatnonzero(roles[fixed]=='U_s')[:2]));fixed=fixed[sel];n=len(fixed);assert n==4"
    assert inner.count(old) == 1
    inner = inner.replace(old, new)
compile(inner, '<source-snr-diagnostic>', 'exec')
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=900)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=920)
stdout = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(s for s in stdout.splitlines() if not s.startswith('SOURCE_SNR_DIAGNOSTIC ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
line = next(s[len('SOURCE_SNR_DIAGNOSTIC '):] for s in stdout.splitlines() if s.startswith('SOURCE_SNR_DIAGNOSTIC '))
result = json.loads(line)
assert result['U_s_labels_read'] == result['V_reads'] == result['test_IQ_reads'] == 0 and result['predictions_before_source_tx_labels']
for view in result['results'].values():
    assert sum(x['count'] for x in view['snr_bins']) == result['source_ids']
    assert sum(x['valid_count'] for x in view['snr_bins']) == view['overall']['valid_count']
if smoke:
    assert result['source_ids'] == 4
    print('VERIFIED_SOURCE_SNR_SMOKE', result['source_ids'])
    raise SystemExit(0)
dest = Path(__file__).parent / 'phase1_source_snr_diagnostic.json'
if dest.exists():
    assert json.loads(dest.read_text(encoding='utf8')) == result
else:
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print('VERIFIED source SNR diagnostic', dest)
