"""Source-only blind-gated phase-equivariant DURA waveform averaging."""
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
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);rx=np.load(cache/'train/rx.npy',allow_pickle=False);day=np.load(cache/'train/day.npy',allow_pickle=False)
q_rng=np.random.default_rng(2026110251);h_rng=np.random.default_rng(2026110351);guard_rng=np.random.default_rng(2026110361);utility_rng=np.random.default_rng(2026110371);cycle_rng=np.random.default_rng(2026110411);snr_rng=np.random.default_rng(2026110451);gated_rng=np.random.default_rng(2026110461);new_rng=np.random.default_rng(2026110471);groups=[]
for role,q_count,h_count,guard_count,take,snr_count in (('L_s',2048,512,2048,1024,2048),('U_s',6144,1536,6144,3072,6144)):
 all_ids=np.flatnonzero(roles==role);q=q_rng.choice(all_ids,size=q_count,replace=False);h=h_rng.choice(all_ids,size=h_count,replace=False);without_h=np.setdiff1d(all_ids,np.sort(h),assume_unique=True);guard=guard_rng.choice(without_h,size=guard_count,replace=False);excluded=np.unique(np.concatenate((q,h,guard)));available=np.setdiff1d(all_ids,excluded,assume_unique=True);utility=utility_rng.choice(available,size=take,replace=False);available=np.setdiff1d(available,np.sort(utility),assume_unique=True);cycle=cycle_rng.choice(available,size=take,replace=False);snr=snr_rng.choice(all_ids,size=snr_count,replace=False);available=np.setdiff1d(all_ids,np.unique(np.concatenate((cycle,snr))),assume_unique=True);gated=gated_rng.choice(available,size=snr_count,replace=False);available=np.setdiff1d(available,np.sort(gated),assume_unique=True);assert len(available)>=take;fresh=np.sort(new_rng.choice(available,size=take,replace=False));assert not np.intersect1d(fresh,cycle).size and not np.intersect1d(fresh,snr).size and not np.intersect1d(fresh,gated).size;groups.append(fresh)
fixed=np.concatenate(groups);assert len(fixed)==len(np.unique(fixed))==4096;source_rx=rx[fixed];source_day=day[fixed];split=np.full(len(fixed),-1,dtype=np.int64);split[np.isin(source_rx,[4,6])&(source_day==3)]=1;split[(source_rx==8)&(source_day==3)]=2;keep=split>0;fixed=fixed[keep];source_rx=source_rx[keep];source_day=source_day[keep];split=split[keep];n=len(fixed);assert n>300
views=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban');clean_iq=np.asarray(clean[fixed],dtype=np.float32);clean_complex=clean_iq[:,0].astype(np.float64)+1j*clean_iq[:,1];phases=2*np.pi*np.arange(8)/8;alphas=(.25,.5,1.);data={}
def wide_mse(wave,truth):
 pad=np.pad(wave,((0,0),(64,64)),mode='reflect');target=truth[:,64:-64];best=np.full(len(wave),np.inf)
 for shift in range(-64,65):
  part=pad[:,64+shift:64+shift+256][:,64:-64];gain=np.sum(np.conj(part)*target,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-target)**2,axis=1);best=np.minimum(best,mse)
 return best
for view in views:
 snrs=[]
 if view=='clean':received=clean_iq;snrs=[100.]*n
 else:
  cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');received=[]
  for j in fixed:
   pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110473,session);v=np.asarray(clean[j],dtype=np.float64);signal,_=normalize_input(v[0]+1j*v[1]);result,meta=ResidualChannel(cfg,stable_seed(2026110472,'source-equivariant-dura-2026110472',view,pid),hw).process(signal);assert np.isfinite(result).all() and not meta['channel_equalization_applied'];received.append(np.stack((result.real,result.imag)).astype(np.float32));snrs.append(float(meta['signal_to_added_noise_db']))
  received=np.stack(received)
 anchors=[];rawp=[];basep=[];averages=[];disagreement=[]
 with torch.no_grad():
  for start in range(0,n,32):
   x=core.tensor(received[start:start+32],'cpu');a=dura(x);anchors.append(a.cpu());rawp.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu());basep.append(F.softmax(teacher.classify_features(teacher.features(a)),1).cpu());z=torch.complex(x[:,0],x[:,1]);parts=[]
   for phase in phases:
    w=z*complex(np.cos(phase),np.sin(phase));rotated=torch.stack((w.real,w.imag),1);out=dura(rotated);back=torch.complex(out[:,0],out[:,1]);parts.append(back)
   stack=torch.stack(parts,1);assert torch.allclose(stack[:,0],torch.complex(a[:,0],a[:,1]),rtol=1e-5,atol=1e-6);mean=stack.mean(1);averages.append(torch.stack((mean.real,mean.imag),1).cpu());disagreement.append(((stack-mean[:,None]).abs().square().mean((1,2))/mean.abs().square().mean(1).clamp_min(1e-12)).sqrt().cpu())
 anchor=np.from_dlpack(torch.cat(anchors).contiguous());avg=np.from_dlpack(torch.cat(averages).contiguous());rawp=np.from_dlpack(torch.cat(rawp).contiguous());basep=np.from_dlpack(torch.cat(basep).contiguous());variation=np.from_dlpack(torch.cat(disagreement).contiguous());raw=received[:,0].astype(np.float64)+1j*received[:,1];base=anchor[:,0].astype(np.float64)+1j*anchor[:,1];mean=avg[:,0].astype(np.float64)+1j*avg[:,1];power=np.mean(abs(raw[:,64:-64])**2,axis=1).clip(min=1e-20);fraction=np.clip(wide_mse(base,raw)/power,1e-8,1.);proxy=10*np.log10(np.maximum(1-fraction,1e-8)/fraction);ratio=np.sqrt(np.sum((received-anchor)**2,axis=(1,2))/np.maximum(np.sum(received**2,axis=(1,2)),1e-12));gate=(ratio>.2)&(proxy<10.);candidates=[];probs=[]
 for alpha in alphas:
  out=base+alpha*gate[:,None]*(mean-base);wiq=np.stack((out.real,out.imag),1).astype(np.float32);candidates.append(out);pred=[]
  with torch.no_grad():
   for start in range(0,n,64):
    x=core.tensor(wiq[start:start+64],'cpu');pred.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu())
  probs.append(np.from_dlpack(torch.cat(pred).contiguous()))
 data[view]=dict(rawp=rawp,basep=basep,anchor=base,probs=np.stack(probs),candidates=np.stack(candidates),gate=gate,proxy=proxy,snr=np.asarray(snrs),variation=variation);print('EQUIVARIANT_SOURCE_CANDIDATES_FIXED',view,flush=True)
print('ALL_EQUIVARIANT_SOURCE_CANDIDATES_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=data['clean']['rawp'];teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target)
def evaluate(i,part,group=None):
 mask=(split==part)&valid;cm=split==part
 if group is not None:mask &= source_rx==group;cm &= source_rx==group
 per={};clean_net=0
 for view in views:
  v=data[view];base=v['basep'].argmax(1);out=v['probs'][i].argmax(1);old=base==target;new=out==target;lm=mask&labeled
  per[view]=dict(valid_count=int(mask.sum()),L_s_count=int(lm.sum()),U_confident_count=int((mask&~labeled).sum()),baseline=int((mask&old).sum()),student=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()),gate_enabled=int((cm&v['gate']).sum()),median_orbit_variation=float(np.median(v['variation'][cm])) if cm.any() else None)
  if view=='clean':clean_net=int((cm&(out==teacher_target)).sum()-(cm&(base==teacher_target)).sum())
 return per,clean_net
validation=[];best={'kind':'dura','validation_net':0}
for i,alpha in enumerate(alphas):
 per,clean_net=evaluate(i,1);changes={v:per[v]['student']-per[v]['baseline'] for v in views[1:]};net=sum(changes.values());groups_v={}
 for group in (4,6):
  gp,gclean=evaluate(i,1,group);gc={v:gp[v]['student']-gp[v]['baseline'] for v in views[1:]};groups_v[str(group)]=dict(n=int(((split==1)&(source_rx==group)).sum()),net=sum(gc.values()),per_view=gc,clean_teacher_net=gclean)
 validation.append(dict(candidate=i,alpha=alpha,net=net,per_view=changes,clean_teacher_net=clean_net,groups=groups_v));guards=all(z['net']>=0 and z['clean_teacher_net']>=0 and min(z['per_view'].values())>=-.005*z['n'] for z in groups_v.values())
 if net>best['validation_net'] and guards:best=dict(kind='phase_invariant_dura',candidate=i,alpha=alpha,validation_net=net)
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
print('SOURCE_INVARIANT_DURA '+json.dumps(dict(status='SOURCE_ONLY_INVARIANT_DURA',source_ids=n,validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),phase_count=8,gate_db=10.,alphas=alphas,validation=validation,selected=best,heldout=heldout,heldout_clean_teacher_net=heldout_clean,quality=quality,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,candidates_before_source_tx_labels=True,
 caveat='Every candidate averages DURA outputs in the observed common output frame without inverse rotation; this design follows inspection of these same source records and is post-hoc; inference uses only received IQ and frozen DURA, not source clean IQ or simulator SNR. The blind gate threshold was chosen on earlier source data. Fresh IDs exclude the earlier cycle-network, noise-proxy, and gated-evaluation samples but may overlap other diagnostics; frozen experts saw source records. Simulated satellite is not a validated real link.'),ensure_ascii=False),flush=True)
'''
smoke = '--smoke' in sys.argv
if smoke:
    old = 'n=len(fixed);assert n>300'
    new = old + ";sel=np.concatenate((np.flatnonzero(source_rx==4)[:1],np.flatnonzero(source_rx==6)[:1],np.flatnonzero(source_rx==8)[:2]));fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel];n=len(fixed);assert n==4"
    assert inner.count(old) == 1
    inner = inner.replace(old, new)
    old = "for view in views:\n snrs=[]"
    new = "test=np.array([1+2j,-.5+.2j]);direct=np.mean(np.stack([test for t in phases]),axis=0);wrong=np.mean(np.stack([test*np.exp(-1j*t) for t in phases]),axis=0);assert np.max(abs(direct-test))<1e-12 and np.max(abs(wrong))<1e-12\n" + old
    assert inner.count(old) == 1
    inner = inner.replace(old, new)
compile(inner, '<source-equivariant-dura>', 'exec')
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
marker = 'SOURCE_INVARIANT_DURA '
payload = None
for line in proc.stdout:
    if line.startswith(marker):
        payload = line[len(marker):]
    else:
        print(line, end='', flush=True)
rc = proc.wait()
if rc == 0:
    if payload is None:
        raise ValueError('source equivariant result marker missing')
    result = json.loads(payload)
    assert result['U_s_labels_read'] == result['V_reads'] == result['test_IQ_reads'] == 0 and result['candidates_before_source_tx_labels']
    if smoke:
        assert result['source_ids'] == 4
        print('VERIFIED_SOURCE_INVARIANT_DURA_SMOKE', result['source_ids'], flush=True)
        raise SystemExit(0)
    dest = Path(__file__).parent / 'phase1_source_invariant_dura.json'
    if dest.exists():
        assert json.loads(dest.read_text(encoding='utf8')) == result
    else:
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
    print('SAVED_SOURCE_INVARIANT_DURA', dest, flush=True)
raise SystemExit(rc)
