"""Source-only contrastive ground-memory retrieval network."""
import ast,json,subprocess,sys
from pathlib import Path
p=Path(__file__).parent
tree=ast.parse((p/'phase1_source_phase_utility_probe.py').read_text(encoding='utf8'))
inner=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='inner' for t in n.targets)))
a=inner.index('q_rng=np.random.default_rng(2026110251)');b=inner.index('fixed=np.concatenate(groups)',a)
inner=inner[:a]+'''rng=np.random.default_rng(2026110501);groups=[]\nfor role,take in (("L_s",1024),("U_s",3072)):\n available=np.flatnonzero(roles==role);groups.append(np.sort(rng.choice(available,size=take,replace=False)))\n'''+inner[b:]
inner=inner.split("views=('clean','practical_low_suburban'",1)[0]
if '--smoke' in sys.argv:
 inner=inner.replace('assert all(int((split==i).sum())>200 for i in range(3))','assert all(int((split==i).sum())>200 for i in range(3));sel=np.concatenate([np.flatnonzero(split==i)[:48] for i in range(3)]);fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel]')
tree=ast.parse((p/'phase1_source_ground_memory_probe.py').read_text(encoding='utf8'))
extra=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='inner'))
inner+=extra.split('# Cosine kNN and class centroids.',1)[0]
inner+=r'''
# Learn one shared embedding for degraded queries and clean training references.
# Use only training records and their L_s/teacher-pseudo memory targets.
torch.manual_seed(2026110511);d=bank.shape[1]
class Embed(torch.nn.Module):
 def __init__(self):
  super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(d,128),torch.nn.GELU(),torch.nn.Linear(128,48))
 def forward(self,x):return F.normalize(self.net(x),dim=1)
model=Embed();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001);rng=np.random.default_rng(2026110512)
train_clean=core.tensor(bank.astype(np.float32),'cpu');train_queries=torch.stack([core.tensor(data[v]['features'][bank_ids].astype(np.float32),'cpu') for v in views[1:]])
train_y=torch.tensor(bank_labels.tolist(),dtype=torch.long);epochs=(2,5,10,20);saved={};losses=[]
for epoch in range(1,21):
 model.train();order=rng.permutation(len(bank_ids));tot=0.;num=0
 for start in range(0,len(order),128):
  q=np.asarray(order[start:start+128]);refs=rng.choice(len(bank_ids),size=min(256,len(bank_ids)),replace=False);qix=torch.tensor(q.tolist(),dtype=torch.long);rix=torch.tensor(refs.tolist(),dtype=torch.long);vi=torch.tensor(rng.integers(0,3,size=len(q)).tolist(),dtype=torch.long)
  qz=model(train_queries[vi,qix]);rz=model(train_clean[rix]);logits=qz@rz.T/.10;positive=train_y[qix,None]==train_y[rix][None,:]
  assert positive.any(1).all();contrast=torch.logsumexp(logits,dim=1)-torch.logsumexp(logits.masked_fill(~positive,-torch.inf),dim=1);pair=1-(qz*model(train_clean[qix])).sum(1);loss=(contrast+.1*pair).mean()
  if not torch.isfinite(loss):raise FloatingPointError('finite retrieval loss required')
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();tot+=float(loss)*len(q);num+=len(q)
 losses.append(tot/num)
 if epoch in epochs:saved[epoch]={k:v.detach().clone() for k,v in model.state_dict().items()};print('CONTRASTIVE_MEMORY_EPOCH',epoch,flush=True)
# The source validation labels are used solely for selection; the receiver-8
# holdout labels are not inspected until a configuration has been selected.
labels=fs.source_labels(root,cache,clean,roles,contract)[fixed];target=np.where(labeled,labels,rawp.argmax(1));valid=labeled|(rawp.max(1)>=.95)
ks=(1,5,15,30);preds={}
for epoch in epochs:
 model.load_state_dict(saved[epoch]);model.eval()
 with torch.no_grad():
  memory=np.from_dlpack(model(train_clean).contiguous());memory_centers=np.stack([memory[bank_labels==c].mean(0) for c in range(6)]);memory_centers/=np.maximum(np.linalg.norm(memory_centers,axis=1,keepdims=True),1e-12)
  for view in views:
   emb=np.from_dlpack(model(core.tensor(data[view]['features'].astype(np.float32),'cpu')).contiguous());sim=emb@memory.T;nearest=np.argsort(-sim,axis=1)[:,:max(ks)];choices={'centroid':(emb@memory_centers.T).argmax(1)}
   for k in ks:
    near=bank_labels[nearest[:,:k]];choices[str(k)]=np.stack([(near==c).sum(1) for c in range(6)],1).argmax(1)
   preds[epoch,view]=choices
 print('CONTRASTIVE_MEMORY_PREDICTIONS_FIXED',epoch,flush=True)
def evaluate(epoch,method,part,group=None):
 mask=(split==part)&valid
 if group is not None:mask &= source_rx==group
 out={}
 for view in views:
  base=data[view]['probs'].argmax(1);pred=preds[epoch,view][method];old=base==target;new=pred==target;lm=mask&labeled
  out[view]=dict(count=int(mask.sum()),L_s_count=int(lm.sum()),baseline=int((mask&old).sum()),candidate=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_candidate=int((lm&new).sum()))
 return out
validation=[];best=dict(kind='dura',validation_net=0)
for epoch in epochs:
 for method in ('1','5','15','30','centroid'):
  per=evaluate(epoch,method,1);changes={v:per[v]['candidate']-per[v]['baseline'] for v in views};net=sum(changes[v] for v in views[1:]);groups={}
  for group in (4,6):
   gp=evaluate(epoch,method,1,group);groups[str(group)]={v:gp[v]['candidate']-gp[v]['baseline'] for v in views}
  validation.append(dict(epoch=epoch,method=method,net=net,per_view=changes,groups=groups));guard=all(sum(g[v] for v in views[1:])>=0 and g['clean']>=0 and min(g[v] for v in views[1:])>=-1 for g in groups.values())
  if net>best['validation_net'] and guard:best=dict(kind='contrastive_ground_memory',epoch=epoch,method=method,validation_net=net)
heldout=evaluate(best['epoch'],best['method'],2) if best['kind']!='dura' else None
print('SOURCE_CONTRASTIVE_MEMORY '+json.dumps(dict(status='SOURCE_ONLY_CONTRASTIVE_MEMORY',source_ids=n,train_ids=int((split==0).sum()),validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),bank_ids=int(len(bank_ids)),losses=losses,validation=validation,selected=best,heldout=heldout,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,caveat='Only source training IDs and L_s/confident-teacher pseudo-targets fit this embedding. Source validation selects epoch and retrieval rule; receiver-8 source holdout is evaluated after selection. This is class retrieval, not a signal restoration result. Frozen teacher and DURA saw source records; simulator is not validated real satellite data.'),ensure_ascii=False),flush=True)
'''
compile(inner,'<source-contrastive-memory>','exec')
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
 if line.startswith('SOURCE_CONTRASTIVE_MEMORY '):payload=line[len('SOURCE_CONTRASTIVE_MEMORY '):]
 else:print(line,end='',flush=True)
rc=proc.wait()
if rc:raise SystemExit(rc)
if payload is None:raise ValueError('missing result')
res=json.loads(payload);assert res['U_s_labels_read']==res['V_reads']==res['test_IQ_reads']==0
if '--smoke' in sys.argv:print('VERIFIED_SOURCE_CONTRASTIVE_MEMORY_SMOKE',res['source_ids']);raise SystemExit(0)
dest=p/'phase1_source_contrastive_memory.json'
if dest.exists():assert json.loads(dest.read_text(encoding='utf8'))==res
else:dest.write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
print('SAVED_SOURCE_CONTRASTIVE_MEMORY',dest,flush=True)
