"""Source-only transductive prototype selector, using one receiver/view batch."""
import ast
import json
import subprocess
import sys
from pathlib import Path

p = Path(__file__).parent
tree = ast.parse((p / 'phase1_source_cycle_projection_probe.py').read_text(encoding='utf8'))
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
old = 'teacher,contract,dura,*_=models'
new = 'teacher,contract,dura,boundary,feature,_,_,_=models'
assert inner.count(old) == 1
inner = inner.replace(old, new)
marker = "views=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban')"
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + r'''
views=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban');data={};clean_iq=np.asarray(clean[fixed],dtype=np.float32);clean_complex=clean_iq[:,0].astype(np.float64)+1j*clean_iq[:,1]
for view in views:
 if view=='clean':received=clean_iq
 else:
  cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');received=[]
  for j in fixed:
   pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110426,session);v=np.asarray(clean[j],dtype=np.float64);signal,_=normalize_input(v[0]+1j*v[1]);result,meta=ResidualChannel(cfg,stable_seed(2026110425,'source-batch-prototype-2026110425',view,pid),hw).process(signal)
   assert np.isfinite(result).all() and not meta['channel_equalization_applied'];received.append(np.stack((result.real,result.imag)).astype(np.float32))
  received=np.stack(received)
 probs=[];features=[];waves=[];ratio=[]
 with torch.no_grad():
  for start in range(0,n,32):
   x=core.tensor(received[start:start+32],'cpu');anchor=dura(x);fa=teacher.features(anchor);base=teacher.classify_features(fa);arms=(x,anchor,boundary(x,anchor),feature(x,anchor,fa,base));emb=(teacher.features(arms[0]),fa,teacher.features(arms[2]),teacher.features(arms[3]));probs.append(torch.stack([F.softmax(teacher.classify_features(z),1) for z in emb],1).cpu());features.append(torch.stack([F.normalize(z,dim=1) for z in emb],1).cpu());waves.append(torch.stack(arms,1).cpu());ratio.append(((x-anchor).square().sum((1,2))/x.square().sum((1,2)).clamp_min(1e-12)).sqrt().cpu())
 data[view]=dict(prob=np.from_dlpack(torch.cat(probs).contiguous()),feature=np.from_dlpack(torch.cat(features).contiguous()),wave=np.from_dlpack(torch.cat(waves).contiguous()),ratio=np.from_dlpack(torch.cat(ratio).contiguous()));print('BATCH_PROTOTYPE_SOURCE_OUTPUTS_FIXED',view,flush=True)
seed_thresholds=(.9,.95,.99);margins=(0.,.005,.01,.02,.05);candidates={}
for seed_threshold in seed_thresholds:
 for margin in margins:
  key=f'{seed_threshold}:{margin}';candidates[key]={}
  for view in views:
   v=data[view];p=v['prob'];emb=v['feature'][:,1,:];base=p[:,1,:].argmax(1);conf=p[:,1,:].max(1);all_pred=p.argmax(2);choice=np.ones(n,dtype=np.int64);proto_class=np.full(n,-1,dtype=np.int64);proto_margin=np.full(n,-np.inf)
   for group in (4,6,8):
    ix=np.flatnonzero(source_rx==group);assert len(ix)>10;local=emb[ix];local_base=base[ix];local_conf=conf[ix];seeds=local_conf>=seed_threshold;sums=np.stack([local[seeds&(local_base==c)].sum(0) for c in range(6)]);counts=np.array([int((seeds&(local_base==c)).sum()) for c in range(6)]);numer=local@sums.T;norm2=np.broadcast_to(np.sum(sums*sums,axis=1),(len(ix),6)).copy();cnt=np.broadcast_to(counts,(len(ix),6)).copy();rows=np.flatnonzero(seeds);classes=local_base[rows];numer[rows,classes]-=np.sum(local[rows]*local[rows],axis=1);norm2[rows,classes]-=2*np.sum(local[rows]*sums[classes],axis=1)-np.sum(local[rows]*local[rows],axis=1);cnt[rows,classes]-=1
    for row in rows[:3]:
     c=local_base[row];direct=sums[c]-local[row];assert np.isclose(norm2[row,c],np.sum(direct*direct),atol=1e-4) and np.isclose(numer[row,c],np.dot(local[row],direct),atol=1e-4)
    scores=numer/np.sqrt(np.maximum(norm2,1e-20));scores[cnt<3]=-np.inf;ordered=np.sort(scores,axis=1);has=np.isfinite(ordered[:,-2]);q=scores.argmax(1);m=np.full(len(ix),-np.inf);m[has]=ordered[has,-1]-ordered[has,-2];proto_class[ix[has]]=q[has];proto_margin[ix[has]]=m[has]
   valid_proto=proto_class>=0;safe=np.maximum(proto_class,0);desired=np.broadcast_to(safe[:,None,None],(n,4,1));candidate_prob=np.take_along_axis(p,desired,axis=2)[:,:,0].copy();candidate_prob[all_pred!=safe[:,None]]=-np.inf;candidate_prob[:,1]=-np.inf;arm=candidate_prob.argmax(1);has=np.isfinite(candidate_prob[np.arange(n),arm]);use=valid_proto&has&(proto_class!=base)&(proto_margin>=margin)&(conf<.99)&(v['ratio']>.2);choice[use]=arm[use];candidates[key][view]=dict(choice=choice,proto_class=proto_class,proto_margin=proto_margin)
 print('BATCH_PROTOTYPE_SOURCE_DECISIONS_FIXED',seed_threshold,flush=True)
print('ALL_BATCH_PROTOTYPE_DECISIONS_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=data['clean']['prob'][:,0,:];teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target)
def evaluate(key,part,group=None):
 mask=(split==part)&valid;cm=split==part
 if group is not None:mask &= source_rx==group;cm &= source_rx==group
 per={};clean_net=0
 for view in views:
  v=data[view];d=candidates[key][view];base=v['prob'][:,1,:].argmax(1);all_pred=v['prob'].argmax(2);out=all_pred[np.arange(n),d['choice']];old=base==target;new=out==target;lm=mask&labeled
  per[view]=dict(valid_count=int(mask.sum()),L_s_count=int(lm.sum()),U_confident_count=int((mask&~labeled).sum()),baseline=int((mask&old).sum()),student=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()),choices=[int((mask&(d['choice']==j)).sum()) for j in range(4)],prototype_correct=int((mask&(d['proto_class']==target)).sum()),prototype_available=int((mask&(d['proto_class']>=0)).sum()))
  if view=='clean':clean_net=int((cm&(out==teacher_target)).sum()-(cm&(base==teacher_target)).sum())
 return per,clean_net
validation=[];best={'kind':'dura','validation_net':0}
for threshold in seed_thresholds:
 for margin in margins:
  key=f'{threshold}:{margin}';per,clean_net=evaluate(key,1);changes={v:per[v]['student']-per[v]['baseline'] for v in views[1:]};net=sum(changes.values());groups_v={}
  for group in (4,6):
   gp,gclean=evaluate(key,1,group);gc={v:gp[v]['student']-gp[v]['baseline'] for v in views[1:]};groups_v[str(group)]=dict(n=int(((split==1)&(source_rx==group)).sum()),net=sum(gc.values()),per_view=gc,clean_teacher_net=gclean)
  validation.append(dict(seed_threshold=threshold,margin=margin,net=net,per_view=changes,clean_teacher_net=clean_net,groups=groups_v))
  guards=all(z['net']>=0 and z['clean_teacher_net']>=0 and min(z['per_view'].values())>=-.005*z['n'] for z in groups_v.values())
  if net>best['validation_net'] and guards:best=dict(kind='batch_prototype_selector',seed_threshold=threshold,margin=margin,validation_net=net)
def aligned_mse(wave,truth):
 pad=np.pad(wave,((0,0),(16,16)),mode='reflect');truth=truth[:,16:-16];best=np.full(len(wave),np.inf)
 for shift in range(-16,17):
  part=pad[:,16+shift:16+shift+256][:,16:-16];gain=np.sum(np.conj(part)*truth,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-truth)**2,axis=1);best=np.minimum(best,mse)
 return best
if best['kind']=='dura':heldout=None;heldout_clean=None;quality=None
else:
 key=f"{best['seed_threshold']}:{best['margin']}";heldout,heldout_clean=evaluate(key,2);mask=split==2;quality={}
 for view in views:
  v=data[view];d=candidates[key][view];w=v['wave'];out=w[np.arange(n),d['choice']];anchor=w[:,1];after=out[:,0].astype(np.float64)+1j*out[:,1];before=anchor[:,0].astype(np.float64)+1j*anchor[:,1];b=aligned_mse(before[mask],clean_complex[mask]);a=aligned_mse(after[mask],clean_complex[mask]);quality[view]=dict(count=int(mask.sum()),aligned_mse_before=float(b.mean()),aligned_mse_after=float(a.mean()),improved=int((a<b).sum()),worsened=int((a>b).sum()))
print('SOURCE_BATCH_PROTOTYPE '+json.dumps(dict(status='SOURCE_ONLY_BATCH_PROTOTYPE_PROBE',source_ids=n,validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),seed_thresholds=seed_thresholds,margins=margins,validation=validation,selected=best,heldout=heldout,heldout_clean_teacher_net=heldout_clean,quality=quality,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,decisions_before_source_tx_labels=True,
 caveat='Transductive batch inference uses other unlabeled waveforms from the same receiver/day and same channel view; no matched clean or cross-view IQ is used for inference. Each record is excluded from its own prototype. Receiver grouping and a sufficiently large batch are assumed. Source IDs overlap prior cycle diagnostics, and frozen experts saw source data. Simulated satellite is not validated real satellite; earlier Phase1 test results motivated exploration.'),ensure_ascii=False),flush=True)
'''
smoke = '--smoke' in sys.argv
if smoke:
    old = 'n=len(fixed);assert n>300'
    new = old + ";sel=np.concatenate([np.flatnonzero(source_rx==g)[:12] for g in (4,6,8)]);fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel];n=len(fixed);assert n==36"
    assert inner.count(old) == 1
    inner = inner.replace(old, new)
compile(inner, '<source-batch-prototype>', 'exec')
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=1800)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=1820)
stdout = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(s for s in stdout.splitlines() if not s.startswith('SOURCE_BATCH_PROTOTYPE ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
line = next(s[len('SOURCE_BATCH_PROTOTYPE '):] for s in stdout.splitlines() if s.startswith('SOURCE_BATCH_PROTOTYPE '))
result = json.loads(line)
assert result['U_s_labels_read'] == result['V_reads'] == result['test_IQ_reads'] == 0 and result['decisions_before_source_tx_labels']
if smoke:
    assert result['source_ids'] == 36 and len(result['validation']) == 15
    print('VERIFIED_SOURCE_BATCH_PROTOTYPE_SMOKE', result['source_ids'])
    raise SystemExit(0)
dest = p / 'phase1_source_batch_prototype_probe.json'
if dest.exists():
    assert json.loads(dest.read_text(encoding='utf8')) == result
else:
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print('SAVED_SOURCE_BATCH_PROTOTYPE', dest)
