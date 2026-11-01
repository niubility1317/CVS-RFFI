"""Source-only phase-orbit selection using DURA's internal forward residual."""
import ast
import json
import subprocess
import sys
from pathlib import Path

p = Path(__file__).parent
tree = ast.parse((p / 'phase1_source_equivariant_dura.py').read_text(encoding='utf8'))
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
start = inner.index('q_rng=np.random.default_rng(2026110251)')
end = inner.index('fixed=np.concatenate(groups)', start)
sampling = r'''q_rng=np.random.default_rng(2026110251);h_rng=np.random.default_rng(2026110351);guard_rng=np.random.default_rng(2026110361);utility_rng=np.random.default_rng(2026110371);cycle_rng=np.random.default_rng(2026110411);snr_rng=np.random.default_rng(2026110451);gated_rng=np.random.default_rng(2026110461);equiv_rng=np.random.default_rng(2026110471);new_rng=np.random.default_rng(2026110481);groups=[]
for role,q_count,h_count,guard_count,take,snr_count in (('L_s',2048,512,2048,1024,2048),('U_s',6144,1536,6144,3072,6144)):
 all_ids=np.flatnonzero(roles==role);q=q_rng.choice(all_ids,size=q_count,replace=False);h=h_rng.choice(all_ids,size=h_count,replace=False);without_h=np.setdiff1d(all_ids,np.sort(h),assume_unique=True);guard=guard_rng.choice(without_h,size=guard_count,replace=False);excluded=np.unique(np.concatenate((q,h,guard)));available=np.setdiff1d(all_ids,excluded,assume_unique=True);utility=utility_rng.choice(available,size=take,replace=False);available=np.setdiff1d(available,np.sort(utility),assume_unique=True);cycle=cycle_rng.choice(available,size=take,replace=False);snr=snr_rng.choice(all_ids,size=snr_count,replace=False);available=np.setdiff1d(all_ids,np.unique(np.concatenate((cycle,snr))),assume_unique=True);gated=gated_rng.choice(available,size=snr_count,replace=False);available=np.setdiff1d(available,np.sort(gated),assume_unique=True);equiv=equiv_rng.choice(available,size=take,replace=False);available=np.setdiff1d(all_ids,np.unique(np.concatenate((equiv,snr))),assume_unique=True);assert len(available)>=snr_count;fresh=np.sort(new_rng.choice(available,size=snr_count,replace=False));assert not np.intersect1d(fresh,equiv).size and not np.intersect1d(fresh,snr).size;groups.append(fresh)
'''
inner = inner[:start] + sampling + inner[end:]
assert inner.count('len(fixed)==len(np.unique(fixed))==4096') == 1
inner = inner.replace('len(fixed)==len(np.unique(fixed))==4096', 'len(fixed)==len(np.unique(fixed))==8192')
inner = inner.replace('2026110473', '2026110483').replace('2026110472', '2026110482').replace('source-equivariant-dura-2026110482', 'source-dura-action-selector-2026110482')
marker = "views=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban')"
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + r'''
views=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban');clean_iq=np.asarray(clean[fixed],dtype=np.float32);clean_complex=clean_iq[:,0].astype(np.float64)+1j*clean_iq[:,1];phases=2*np.pi*np.arange(8)/8;thresholds=(0.,.01,.05,.1);data={}
def wide_mse(wave,truth):
 pad=np.pad(wave,((0,0),(64,64)),mode='reflect');target=truth[:,64:-64];best=np.full(len(wave),np.inf)
 for shift in range(-64,65):
  part=pad[:,64+shift:64+shift+256][:,64:-64];gain=np.sum(np.conj(part)*target,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-target)**2,axis=1);best=np.minimum(best,mse)
 return best
for view in views:
 if view=='clean':received=clean_iq
 else:
  cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');received=[]
  for j in fixed:
   pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110483,session);v=np.asarray(clean[j],dtype=np.float64);signal,_=normalize_input(v[0]+1j*v[1]);result,meta=ResidualChannel(cfg,stable_seed(2026110482,'source-dura-action-selector-2026110482',view,pid),hw).process(signal);assert np.isfinite(result).all() and not meta['channel_equalization_applied'];received.append(np.stack((result.real,result.imag)).astype(np.float32))
  received=np.stack(received)
 anchors=[];rawp=[];basep=[];orbits=[];costs=[]
 with torch.no_grad():
  for start in range(0,n,32):
   x=core.tensor(received[start:start+32],'cpu');a=dura(x);anchors.append(a.cpu());rawp.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu());basep.append(F.softmax(teacher.classify_features(teacher.features(a)),1).cpu());z=torch.complex(x[:,0],x[:,1]);parts=[];batch_cost=[]
   for phase in phases:
    w=z*complex(np.cos(phase),np.sin(phase));rotated=torch.stack((w.real,w.imag),1);out,variance,action,_=dura.training_outputs(rotated);prediction=action(torch.complex(out[:,0],out[:,1]));cost=((prediction-w).abs().square()/variance.clamp_min(1e-8)).mean(1);assert torch.isfinite(cost).all();back=torch.complex(out[:,0],out[:,1])*complex(np.cos(phase),-np.sin(phase));parts.append(back);batch_cost.append(cost)
   stack=torch.stack(parts,1);assert torch.allclose(stack[:,0],torch.complex(a[:,0],a[:,1]),rtol=1e-5,atol=1e-6);orbits.append(torch.stack((stack.real,stack.imag),2).cpu());costs.append(torch.stack(batch_cost,1).cpu())
 anchor=np.from_dlpack(torch.cat(anchors).contiguous());orbit=np.from_dlpack(torch.cat(orbits).contiguous());cost=np.from_dlpack(torch.cat(costs).contiguous());rawp=np.from_dlpack(torch.cat(rawp).contiguous());basep=np.from_dlpack(torch.cat(basep).contiguous());raw=received[:,0].astype(np.float64)+1j*received[:,1];base=anchor[:,0].astype(np.float64)+1j*anchor[:,1];power=np.mean(abs(raw[:,64:-64])**2,axis=1).clip(min=1e-20);fraction=np.clip(wide_mse(base,raw)/power,1e-8,1.);proxy=10*np.log10(np.maximum(1-fraction,1e-8)/fraction);ratio=np.sqrt(np.sum((received-anchor)**2,axis=(1,2))/np.maximum(np.sum(received**2,axis=(1,2)),1e-12));gate=(ratio>.2)&(proxy<10.);pred=[]
 with torch.no_grad():
  for phase in range(8):
   out=[]
   for start in range(0,n,64):
    x=core.tensor(orbit[start:start+64,phase],'cpu');out.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu())
   pred.append(np.from_dlpack(torch.cat(out).contiguous()).argmax(1))
 pred=np.stack(pred,1);assert np.array_equal(pred[:,0],basep.argmax(1));data[view]=dict(rawp=rawp,basep=basep,anchor=base,orbit=orbit,pred=pred,cost=cost,gate=gate);print('DURA_ACTION_SOURCE_ORBIT_FIXED',view,flush=True)
candidates={}
for threshold in thresholds:
 per={}
 for view in views:
  v=data[view];idx=v['cost'].argmin(1);best=v['cost'][np.arange(n),idx];base=v['cost'][:,0];use=v['gate']&(best<base*(1-threshold));choice=np.where(use,idx,0);per[view]=choice
 candidates[str(threshold)]=per
print('ALL_DURA_ACTION_DECISIONS_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=data['clean']['rawp'];teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target)
def evaluate(threshold,part,group=None):
 mask=(split==part)&valid;cm=split==part
 if group is not None:mask &= source_rx==group;cm &= source_rx==group
 per={};clean_net=0
 for view in views:
  v=data[view];choice=candidates[str(threshold)][view];base=v['pred'][:,0];out=v['pred'][np.arange(n),choice];old=base==target;new=out==target;lm=mask&labeled
  per[view]=dict(valid_count=int(mask.sum()),L_s_count=int(lm.sum()),U_confident_count=int((mask&~labeled).sum()),baseline=int((mask&old).sum()),student=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()),switches=int((mask&(choice!=0)).sum()),gate_enabled=int((cm&v['gate']).sum()))
  if view=='clean':clean_net=int((cm&(out==teacher_target)).sum()-(cm&(base==teacher_target)).sum())
 return per,clean_net
validation=[];best={'kind':'dura','validation_net':0}
for threshold in thresholds:
 per,clean_net=evaluate(threshold,1);changes={v:per[v]['student']-per[v]['baseline'] for v in views[1:]};net=sum(changes.values());groups_v={}
 for group in (4,6):
  gp,gclean=evaluate(threshold,1,group);gc={v:gp[v]['student']-gp[v]['baseline'] for v in views[1:]};groups_v[str(group)]=dict(n=int(((split==1)&(source_rx==group)).sum()),net=sum(gc.values()),per_view=gc,clean_teacher_net=gclean)
 validation.append(dict(threshold=threshold,net=net,per_view=changes,clean_teacher_net=clean_net,groups=groups_v));guards=all(z['net']>=0 and z['clean_teacher_net']>=0 and min(z['per_view'].values())>=-.005*z['n'] for z in groups_v.values())
 if net>best['validation_net'] and guards:best=dict(kind='dura_action_phase_selector',threshold=threshold,validation_net=net)
def aligned_mse(wave,truth):
 pad=np.pad(wave,((0,0),(16,16)),mode='reflect');truth=truth[:,16:-16];best=np.full(len(wave),np.inf)
 for shift in range(-16,17):
  part=pad[:,16+shift:16+shift+256][:,16:-16];gain=np.sum(np.conj(part)*truth,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-truth)**2,axis=1);best=np.minimum(best,mse)
 return best
if best['kind']=='dura':heldout=None;heldout_clean=None;quality=None
else:
 heldout,heldout_clean=evaluate(best['threshold'],2);mask=split==2;quality={}
 for view in views:
  v=data[view];choice=candidates[str(best['threshold'])][view];out=v['orbit'][np.arange(n),choice];wave=out[:,0].astype(np.float64)+1j*out[:,1];before=aligned_mse(v['anchor'][mask],clean_complex[mask]);after=aligned_mse(wave[mask],clean_complex[mask]);quality[view]=dict(count=int(mask.sum()),aligned_mse_before=float(before.mean()),aligned_mse_after=float(after.mean()),improved=int((after<before).sum()),worsened=int((after>before).sum()))
print('SOURCE_DURA_ACTION_SELECTOR '+json.dumps(dict(status='SOURCE_ONLY_DURA_ACTION_SELECTOR',source_ids=n,validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),phase_count=8,gate_db=10.,thresholds=thresholds,validation=validation,selected=best,heldout=heldout,heldout_clean_teacher_net=heldout_clean,quality=quality,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,decisions_before_source_tx_labels=True,
 caveat='Each candidate uses DURA on a phase-rotated received IQ signal; the per-candidate score is DURA internal estimated forward residual divided by its estimated variance. These are model self-consistency scores, not true channel likelihoods. Inference uses only received IQ and fixed DURA; no source clean IQ, simulator truth, source targets or cross-view signals. Fresh IDs exclude the phase-orbit and proxy sample but may overlap other diagnostics; frozen models saw source records. Simulated satellite is not a validated real link.'),ensure_ascii=False),flush=True)
'''
smoke = '--smoke' in sys.argv
if smoke:
    old = 'n=len(fixed);assert n>300'
    new = old + ";sel=np.concatenate((np.flatnonzero(source_rx==4)[:1],np.flatnonzero(source_rx==6)[:1],np.flatnonzero(source_rx==8)[:2]));fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel];n=len(fixed);assert n==4"
    assert inner.count(old) == 1
    inner = inner.replace(old, new)
compile(inner, '<source-dura-action-selector>', 'exec')
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
marker = 'SOURCE_DURA_ACTION_SELECTOR '
payload = None
for line in proc.stdout:
    if line.startswith(marker):
        payload = line[len(marker):]
    else:
        print(line, end='', flush=True)
rc = proc.wait()
if rc == 0:
    if payload is None:
        raise ValueError('source action-selector result marker missing')
    result = json.loads(payload)
    assert result['U_s_labels_read'] == result['V_reads'] == result['test_IQ_reads'] == 0 and result['decisions_before_source_tx_labels']
    if smoke:
        assert result['source_ids'] == 4
        print('VERIFIED_SOURCE_DURA_ACTION_SMOKE', result['source_ids'], flush=True)
        raise SystemExit(0)
    dest = p / 'phase1_source_dura_action_selector.json'
    if dest.exists():
        assert json.loads(dest.read_text(encoding='utf8')) == result
    else:
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
    print('SAVED_SOURCE_DURA_ACTION_SELECTOR', dest, flush=True)
raise SystemExit(rc)
