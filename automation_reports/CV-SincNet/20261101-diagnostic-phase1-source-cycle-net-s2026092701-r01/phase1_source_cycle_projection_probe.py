"""Source-only phase-aligned 20-sample periodic/residual restoration pilot."""
import json
import subprocess
from pathlib import Path

inner = r'''
from pathlib import Path
import json,numpy as np,torch
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,physical_rich_router_experiment as g,feature_conditioned_experiment as fs
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2);models=g.load_models(root,'cpu');teacher,contract,dura,*_=models;cache,clean,roles=core.data(root,contract)
from leo_practical.channel import Config,receiver_for_session,stable_seed,normalize_input
from leo_practical.residual import ResidualChannel
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);rx=np.load(cache/'train/rx.npy',allow_pickle=False);day=np.load(cache/'train/day.npy',allow_pickle=False)
q_rng=np.random.default_rng(2026110251);h_rng=np.random.default_rng(2026110351);guard_rng=np.random.default_rng(2026110361);utility_rng=np.random.default_rng(2026110371);fresh_rng=np.random.default_rng(2026110411);groups=[]
for role,q_count,h_count,guard_count,take in (('L_s',2048,512,2048,1024),('U_s',6144,1536,6144,3072)):
 all_ids=np.flatnonzero(roles==role);q_ids=q_rng.choice(all_ids,size=q_count,replace=False);h_ids=h_rng.choice(all_ids,size=h_count,replace=False);without_h=np.setdiff1d(all_ids,np.sort(h_ids),assume_unique=True);guard_ids=guard_rng.choice(without_h,size=guard_count,replace=False);excluded=np.unique(np.concatenate((q_ids,h_ids,guard_ids)));available=np.setdiff1d(all_ids,excluded,assume_unique=True);utility=utility_rng.choice(available,size=take,replace=False);available=np.setdiff1d(available,np.sort(utility),assume_unique=True);assert len(available)>=take;groups.append(np.sort(fresh_rng.choice(available,size=take,replace=False)))
fixed=np.concatenate(groups);assert len(fixed)==len(np.unique(fixed))==4096;source_rx=rx[fixed];source_day=day[fixed];split=np.full(len(fixed),-1,dtype=np.int64);split[np.isin(source_rx,[4,6])&(source_day==3)]=1;split[(source_rx==8)&(source_day==3)]=2;keep=split>0;fixed=fixed[keep];source_rx=source_rx[keep];source_day=source_day[keep];split=split[keep];n=len(fixed);assert n>300
views=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban');alphas=(-.3,-.1,-.03,.03,.1,.3);period=20;data={};clean_iq=np.asarray(clean[fixed],dtype=np.float32);clean_complex=clean_iq[:,0].astype(np.float64)+1j*clean_iq[:,1];cycles=np.arange(256)//period;residues=np.arange(256)%period
def project(z):
 phi=np.angle(np.sum(np.conj(z[:,:-period])*z[:,period:],axis=1));demod=z*np.exp(-1j*phi[:,None]*cycles);mean=np.stack([np.mean(demod[:,r::period],axis=1) for r in range(period)],1);out=mean[:,residues]*np.exp(1j*phi[:,None]*cycles);return out,phi
for view in views:
 if view=='clean':received=clean_iq
 else:
  cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');received=[]
  for j in fixed:
   pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110413,session);v=np.asarray(clean[j],dtype=np.float64);signal,_=normalize_input(v[0]+1j*v[1]);result,meta=ResidualChannel(cfg,stable_seed(2026110412,'source-cycle-projection-2026110412',view,pid),hw).process(signal)
   assert np.isfinite(result).all() and not meta['channel_equalization_applied'];received.append(np.stack((result.real,result.imag)).astype(np.float32))
  received=np.stack(received)
 anchors=[];rawp=[];basep=[]
 with torch.no_grad():
  for start in range(0,n,32):
   x=core.tensor(received[start:start+32],'cpu');a=dura(x);anchors.append(a.cpu());rawp.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu());basep.append(F.softmax(teacher.classify_features(teacher.features(a)),1).cpu())
 anchor=np.from_dlpack(torch.cat(anchors).contiguous());rawp=np.from_dlpack(torch.cat(rawp).contiguous());basep=np.from_dlpack(torch.cat(basep).contiguous());z=anchor[:,0].astype(np.float64)+1j*anchor[:,1];cycle,phi=project(z);ratio=np.sqrt(np.sum((received-anchor)**2,axis=(1,2))/np.maximum(np.sum(received**2,axis=(1,2)),1e-12));gate=ratio>.2;candidates=[];probs=[]
 for alpha in alphas:
  candidate=z+alpha*gate[:,None]*(cycle-z);candidate_iq=np.stack((candidate.real,candidate.imag),1).astype(np.float32);candidates.append(candidate);out=[]
  with torch.no_grad():
   for start in range(0,n,64):
    x=core.tensor(candidate_iq[start:start+64],'cpu');out.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu())
  probs.append(np.from_dlpack(torch.cat(out).contiguous()))
 data[view]=dict(anchor=z,rawp=rawp,basep=basep,probs=np.stack(probs),candidates=np.stack(candidates),gate=gate);print('CYCLE_SOURCE_CANDIDATES_FIXED',view,flush=True)
print('ALL_CYCLE_SOURCE_CANDIDATES_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)
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
for i,alpha in enumerate(alphas):
 per,clean_net=evaluate(i,1);changes={v:per[v]['student']-per[v]['baseline'] for v in views[1:]};net=sum(changes.values());groups_v={}
 for group in (4,6):
  gp,gclean=evaluate(i,1,group);gc={v:gp[v]['student']-gp[v]['baseline'] for v in views[1:]};groups_v[str(group)]=dict(n=int(((split==1)&(source_rx==group)).sum()),net=sum(gc.values()),per_view=gc,clean_teacher_net=gclean)
 validation.append(dict(candidate=i,alpha=alpha,net=net,per_view=changes,clean_teacher_net=clean_net,groups=groups_v))
 guards=all(z['net']>=0 and z['clean_teacher_net']>=0 and min(z['per_view'].values())>=-.005*z['n'] for z in groups_v.values())
 if net>best['validation_net'] and guards:best=dict(kind='cycle_residual_projection',candidate=i,alpha=alpha,validation_net=net)
def aligned_mse(wave,truth):
 pad=np.pad(wave,((0,0),(16,16)),mode='reflect');truth=truth[:,16:-16];best=np.full(len(wave),np.inf)
 for shift in range(-16,17):
  part=pad[:,16+shift:16+shift+256][:,16:-16];gain=np.sum(np.conj(part)*truth,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-truth)**2,axis=1);best=np.minimum(best,mse)
 return best
if best['kind']=='dura':heldout=None;heldout_clean=None;quality=None
else:
 heldout,heldout_clean=evaluate(best['candidate'],2);mask=split==2;quality={}
 for view in views:
  v=data[view];before=aligned_mse(v['anchor'][mask],clean_complex[mask]);after=aligned_mse(v['candidates'][best['candidate']][mask],clean_complex[mask]);quality[view]=dict(count=int(mask.sum()),aligned_mse_before=float(before.mean()),aligned_mse_after=float(after.mean()),improved=int((after<before).sum()),worsened=int((after>before).sum()),gate_enabled=int(v['gate'][mask].sum()))
print('SOURCE_CYCLE_PROJECTION '+json.dumps(dict(status='SOURCE_ONLY_CYCLE_PROJECTION_PROBE',source_ids=n,validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),period=period,alphas=alphas,validation=validation,selected=best,heldout=heldout,heldout_clean_teacher_net=heldout_clean,quality=quality,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,candidates_before_source_tx_labels=True,
 caveat='Period 20 was selected after inspecting earlier unlabeled source autocorrelation; this source ID sample excludes four named previous samples but may overlap other probes. Paired clean source is used for simulation and quality only, not per-record inference. Frozen experts saw source records. Source simulator is not validated real satellite data; observed Phase1 test results motivated exploration.'),ensure_ascii=False),flush=True)
'''
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=1800)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.Popen(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf8', errors='replace', bufsize=1)
proc.stdin.write(remote)
proc.stdin.close()
marker = 'SOURCE_CYCLE_PROJECTION '
payload = None
for line in proc.stdout:
    if line.startswith(marker):
        payload = line[len(marker):]
    else:
        print(line, end='', flush=True)
rc = proc.wait()
if rc == 0:
    if payload is None:
        raise ValueError('source cycle result marker missing')
    dest = Path(__file__).parent / 'phase1_source_cycle_projection_probe.json'
    if dest.exists():
        assert json.loads(dest.read_text(encoding='utf8')) == json.loads(payload)
    else:
        dest.write_text(payload, encoding='utf8', newline='\n')
    print('SAVED_SOURCE_CYCLE_RESULT', dest, flush=True)
raise SystemExit(rc)
