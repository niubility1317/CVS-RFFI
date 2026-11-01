"""Create a source-only selector that jointly pools all phase embeddings."""
import ast
from pathlib import Path

p = Path(__file__).parent
source = (p / 'phase1_source_phase_utility_probe.py').read_text(encoding='utf8')
tree = ast.parse(source)
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
old = "('base_feat','base_prob','raw_prob','phase_feat','phase_prob','energy')"
assert inner.count(old) == 1
inner = inner.replace(old, "('base_feat','base_prob','raw_prob','phase_feat','phase_prob','energy','ratio')")
old = "buckets['energy'].append(torch.stack((x.square().mean((1,2)).clamp_min(1e-8).log(),a.square().mean((1,2)).clamp_min(1e-8).log()),1).cpu())"
assert inner.count(old) == 1
inner = inner.replace(old, old + "\n   buckets['ratio'].append(((x-a).square().sum((1,2))/x.square().sum((1,2)).clamp_min(1e-12)).sqrt().cpu())")
marker = "print('ALL_PHASE_UTILITY_SOURCE_CANDIDATES_FIXED_BEFORE_SOURCE_LABELS',flush=True)"
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + marker + r'''
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1)
ref=np.from_dlpack(data['clean']['raw_prob']);teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target);n=len(fixed)
features=[];baselines=[];all_phase=[];ratios=[]
for view in views:
 v=data[view];f=v['phase_feat'];prob=v['phase_prob'];feat=torch.cat((f.mean(1),f.std(1,unbiased=False),v['base_prob'],v['raw_prob'],prob.mean(1),prob.std(1,unbiased=False),v['energy']),1)
 assert feat.shape==(n,346);features.append(feat);baselines.append(v['base_prob'].argmax(1).cpu());all_phase.append(prob.cpu());ratios.append(v['ratio'].cpu())
features=torch.cat(features);base=np.from_dlpack(torch.cat(baselines).contiguous());phase=np.from_dlpack(torch.cat(all_phase).contiguous());ratio=np.from_dlpack(torch.cat(ratios).contiguous());all_target=np.tile(target,len(views));all_split=np.tile(split,len(views));all_valid=np.tile(valid,len(views))
train=np.flatnonzero((all_split==0)&all_valid);assert len(train)>1000
train_t=torch.tensor(train.tolist(),dtype=torch.long);mean=features[train_t].mean(0);std=features[train_t].std(0).clamp_min(1e-4);scaled=((features-mean)/std).clamp(-10,10);target_t=torch.tensor(all_target.tolist(),dtype=torch.long)
torch.manual_seed(2026110381);model=nn.Sequential(nn.Linear(346,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,6));opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);rng=np.random.default_rng(2026110382)
thresholds=(.5,.6,.7,.8,.9,.95,.98);validation=[];best={'kind':'dura','validation_net':0};best_state=None
def score_all():
 model.eval();out=[]
 with torch.no_grad():
  for start in range(0,len(scaled),4096):out.append(F.softmax(model(scaled[start:start+4096]),1).cpu())
 return np.from_dlpack(torch.cat(out).contiguous())
def evaluate(q,threshold,part,group=None):
 desired=q.argmax(1);conf=q.max(1);phase_class=phase.argmax(2);eligible=phase_class==desired[:,None];candidate_prob=np.take_along_axis(phase,desired[:,None,None],axis=2)[:,:,0];candidate_prob=np.where(eligible,candidate_prob,-np.inf);choice=candidate_prob.argmax(1);has=np.isfinite(candidate_prob[np.arange(len(base)),choice]);use=has&(conf>=threshold)&(desired!=base)&(ratio>.2);out=base.copy();out[use]=phase_class[np.arange(len(base))[use],choice[use]]
 mask=(split==part)&valid;cm=split==part
 if group is not None:mask &= source_rx==group;cm &= source_rx==group
 per={};clean_net=0
 for i,view in enumerate(views):
  sl=slice(i*n,(i+1)*n);old=base[sl]==target;new=out[sl]==target;lm=mask&labeled
  per[view]=dict(valid_count=int(mask.sum()),L_s_count=int(lm.sum()),U_confident_count=int((mask&~labeled).sum()),baseline=int((mask&old).sum()),student=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()),switches=int((mask&use[sl]).sum()),selector_correct=int((mask&(desired[sl]==target)).sum()))
  if view=='clean':clean_net=int((cm&(out[sl]==teacher_target)).sum()-(cm&(base[sl]==teacher_target)).sum())
 return per,clean_net
for epoch in range(1,31):
 model.train();order=rng.permutation(train)
 for start in range(0,len(order),256):
  ix=torch.tensor(order[start:start+256].tolist(),dtype=torch.long);loss=F.cross_entropy(model(scaled[ix]),target_t[ix])
  if not torch.isfinite(loss):raise FloatingPointError('finite phase-set loss required')
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
 if epoch in (2,5,10,20,30):
  q=score_all()
  for threshold in thresholds:
   per,clean=evaluate(q,threshold,1);changes={v:per[v]['student']-per[v]['baseline'] for v in views[1:]};net=sum(changes.values());groups_v={}
   for group in (4,6):
    gp,gclean=evaluate(q,threshold,1,group);count=int(((split==1)&(source_rx==group)).sum());gc={v:gp[v]['student']-gp[v]['baseline'] for v in views[1:]};groups_v[str(group)]=dict(n=count,net=sum(gc.values()),per_view=gc,clean_teacher_net=gclean)
   validation.append(dict(epoch=epoch,threshold=threshold,net=net,per_view=changes,clean_teacher_net=clean,groups=groups_v))
   guards=all(z['net']>=0 and z['clean_teacher_net']>=0 and min(z['per_view'].values())>=0 for z in groups_v.values())
   if net>best['validation_net'] and guards:
    best=dict(kind='phase_set_class_selector',epoch=epoch,threshold=threshold,validation_net=net);best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
  print('PHASE_SET_SOURCE_VALIDATION_COMPLETE',epoch,flush=True)
if best_state is None:heldout=None;clean_net=None;state=None
else:
 model.load_state_dict(best_state);heldout,clean_net=evaluate(score_all(),best['threshold'],2);state={k:v.detach().tolist() for k,v in model.state_dict().items()}
print('SOURCE_PHASE_SET_PROBE '+json.dumps(dict(status='SOURCE_ONLY_PHASE_SET_PROBE',source_ids=n,train_ids=int((split==0).sum()),validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),phase_count=12,feature_dim=346,guard_ratio=.2,training_records=int(len(train)),validation=validation,selected=best,heldout=heldout,heldout_clean_teacher_net=clean_net,model_state=state,feature_mean=mean.tolist(),feature_std=std.tolist(),U_s_labels_read=0,V_reads=0,test_IQ_reads=0,predictions_before_source_labels=True,
 caveat='Source records and heldout views were previously inspected in other phase experiments, so this is exploratory, not an independent final evaluation. Phase-set selector uses only received IQ and fixed model embeddings at inference. Selector holdout receiver/day is disjoint from selector fitting but frozen experts saw source records. Simulated satellite is not validated real satellite data; earlier Phase1 test observations motivated exploration.'),ensure_ascii=False),flush=True)
'''
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=3600)
'''.replace('INNER', repr(inner), 1)
wrapper = '''import json,subprocess
from pathlib import Path
remote=REMOTE
proc=subprocess.Popen(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',bufsize=1)
proc.stdin.write(remote);proc.stdin.close();marker='SOURCE_PHASE_SET_PROBE ';payload=None
for line in proc.stdout:
 if line.startswith(marker):payload=line[len(marker):]
 else:print(line,end='',flush=True)
rc=proc.wait()
if rc==0:
 if payload is None:raise ValueError('source result marker missing')
 dest=Path(__file__).parent/'phase1_source_phase_set_selector.json'
 if dest.exists():assert json.loads(dest.read_text(encoding='utf8'))==json.loads(payload)
 else:dest.write_text(payload,encoding='utf8',newline='\\n')
 print('SAVED_SOURCE_PHASE_SET_RESULT',dest,flush=True)
raise SystemExit(rc)
'''.replace('REMOTE', repr(remote), 1)
dest = p / 'phase1_source_phase_set_selector.py'
assert not dest.exists()
compile(inner, '<phase-set-inner>', 'exec')
compile(wrapper, str(dest), 'exec')
dest.write_text(wrapper, encoding='utf8', newline='\n')
print('CREATED source-only phase-set selector', dest)
