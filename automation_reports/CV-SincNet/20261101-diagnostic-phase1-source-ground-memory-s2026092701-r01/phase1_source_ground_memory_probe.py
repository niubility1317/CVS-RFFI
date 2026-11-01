"""Source-only fixed ground-feature memory diagnostic; no test data."""
import ast,json,subprocess,sys
from pathlib import Path
p=Path(__file__).parent
tree=ast.parse((p/'phase1_source_phase_utility_probe.py').read_text(encoding='utf8'))
inner=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='inner' for t in n.targets)))
a=inner.index('q_rng=np.random.default_rng(2026110251)');b=inner.index('fixed=np.concatenate(groups)',a)
inner=inner[:a]+'''rng=np.random.default_rng(2026110501);groups=[]\nfor role,take in (("L_s",1024),("U_s",3072)):\n available=np.flatnonzero(roles==role);groups.append(np.sort(rng.choice(available,size=take,replace=False)))\n'''+inner[b:]
inner=inner.split("views=('clean','practical_low_suburban'",1)[0]
if '--smoke' in sys.argv:
 inner=inner.replace('assert all(int((split==i).sum())>200 for i in range(3))', 'assert all(int((split==i).sum())>200 for i in range(3));sel=np.concatenate([np.flatnonzero(split==i)[:48] for i in range(3)]);fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel]')
inner+=r'''
keep=split>=0;fixed=fixed[keep];source_rx=source_rx[keep];source_day=source_day[keep];split=split[keep];n=len(fixed)
assert len(np.unique(fixed))==n and not np.intersect1d(fixed[split==0],fixed[split>0]).size
views=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban');data={};clean_iq=np.asarray(clean[fixed],dtype=np.float32)
for view in views:
 if view=='clean':received=clean_iq
 else:
  cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');received=[]
  for j in fixed:
   pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110503,session);v=np.asarray(clean[j],dtype=np.float64);sig,_=normalize_input(v[0]+1j*v[1]);result,meta=ResidualChannel(cfg,stable_seed(2026110502,'source-ground-memory-2026110502',view,pid),hw).process(sig);assert np.isfinite(result).all() and not meta['channel_equalization_applied'];received.append(np.stack((result.real,result.imag)).astype(np.float32))
  received=np.stack(received)
 features=[];probs=[];rawp=[]
 with torch.no_grad():
  for start in range(0,n,64):
   x=core.tensor(received[start:start+64],'cpu');a=dura(x);f=teacher.features(a);features.append(F.normalize(f,dim=1).cpu());probs.append(F.softmax(teacher.classify_features(f),1).cpu())
   if view=='clean':
    cf=teacher.features(x);rawp.append(F.softmax(teacher.classify_features(cf),1).cpu())
 data[view]=dict(features=np.from_dlpack(torch.cat(features).contiguous()),probs=np.from_dlpack(torch.cat(probs).contiguous()))
 if view=='clean':data[view]['rawp']=np.from_dlpack(torch.cat(rawp).contiguous())
 print('GROUND_MEMORY_FEATURES_FIXED',view,flush=True)
# A memory contains only earlier-day source training IDs, never evaluation IDs.
rawp=data['clean']['rawp'];labeled=roles[fixed]=='L_s';bank_mask=(split==0)&(labeled|(rawp.max(1)>=.95));bank_ids=np.flatnonzero(bank_mask);source_labels=fs.source_labels(root,cache,clean,roles,contract)[fixed];assert np.all(source_labels[~labeled]==-1)
bank_labels=np.where(labeled[bank_ids],source_labels[bank_ids],rawp[bank_ids].argmax(1));del source_labels
assert len(bank_ids)>30 and set(np.unique(bank_labels).tolist())==set(range(6))
bank=data['clean']['features'][bank_ids];assert np.allclose(np.linalg.norm(bank,axis=1),1,atol=1e-4)
# Cosine kNN and class centroids. Evaluate direct memory classification only;
# it is a feasibility diagnostic, not a reconstructed waveform.
ks=(1,5,15,30);predictions={};coverage={}
centroids=np.stack([bank[bank_labels==c].mean(0) for c in range(6)]);centroids/=np.maximum(np.linalg.norm(centroids,axis=1,keepdims=True),1e-12)
for view in views:
 feat=data[view]['features'];similarity=feat@bank.T;nearest=np.argsort(-similarity,axis=1)[:,:max(ks)];predictions[view]={'centroid':(feat@centroids.T).argmax(1)};coverage[view]={}
 for k in ks:
  nn=bank_labels[nearest[:,:k]];votes=np.stack([(nn==c).sum(1) for c in range(6)],1);predictions[view][str(k)]=votes.argmax(1);coverage[view][str(k)]=nn
print('ALL_GROUND_MEMORY_DECISIONS_FIXED_BEFORE_EVALUATION_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract)[fixed];target=np.where(labeled,labels,rawp.argmax(1));valid=labeled|(rawp.max(1)>=.95);results={}
for view in views:
 results[view]={};base=data[view]['probs'].argmax(1)
 for part in (1,2):
  m=(split==part)&valid;lm=m&labeled;old=base==target;items={}
  for key,pred in predictions[view].items():
   new=pred==target;z=dict(correct=int((m&new).sum()),L_s_correct=int((lm&new).sum()),repairs=int((m&~old&new).sum()),damages=int((m&old&~new).sum()))
   if key!='centroid':
    match=(coverage[view][key]==target[:,None]).any(1);z['class_covered']=int((m&match).sum());z['baseline_errors_covered']=int((m&~old&match).sum())
   items[key]=z
  results[view][str(part)]=dict(valid_count=int(m.sum()),L_s_count=int(lm.sum()),baseline=int((m&old).sum()),L_s_baseline=int((lm&old).sum()),memory=items)
print('SOURCE_GROUND_MEMORY '+json.dumps(dict(status='SOURCE_ONLY_GROUND_MEMORY_FEASIBILITY',source_ids=n,train_ids=int((split==0).sum()),validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),bank_ids=int(len(bank_ids)),bank_class_counts=np.bincount(bank_labels,minlength=6).tolist(),results=results,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,decisions_fixed_before_evaluation_label_use=True,caveat='Ground memory stores source training embeddings and true L_s / confident clean-teacher U_s pseudo-labels. Retrieval is a classification diagnostic, not a waveform restorer. The source label helper loads L_s labels but evaluation labels are discarded before retrieval decisions. Frozen teacher and DURA saw source records. Simulated satellite is not validated real satellite.'),ensure_ascii=False),flush=True)
'''
compile(inner,'<ground-memory>','exec')
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
 if line.startswith('SOURCE_GROUND_MEMORY '):payload=line[len('SOURCE_GROUND_MEMORY '):]
 else:print(line,end='',flush=True)
rc=proc.wait()
if rc:raise SystemExit(rc)
if payload is None:raise ValueError('missing result')
res=json.loads(payload);assert res['U_s_labels_read']==res['V_reads']==res['test_IQ_reads']==0 and res['decisions_fixed_before_evaluation_label_use']
if '--smoke' in sys.argv:print('VERIFIED_SOURCE_GROUND_MEMORY_SMOKE',res['source_ids']);raise SystemExit(0)
dest=p/'phase1_source_ground_memory_probe.json'
if dest.exists():assert json.loads(dest.read_text(encoding='utf8'))==res
else:dest.write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
print('SAVED_SOURCE_GROUND_MEMORY',dest,flush=True)
