"""Independently replay fixed phase choices on selector-held-out source records."""
import ast
import json
import subprocess
from pathlib import Path

p = Path(__file__).parent
result = json.loads((p / 'phase1_source_phase_utility_probe.json').read_text(encoding='utf8'))
assert result['status'] == 'SOURCE_ONLY_PHASE_UTILITY_PROBE'
assert result['selected']['kind'] == 'phase_utility' and result['heldout'] is not None
tree = ast.parse((p / 'phase1_source_phase_utility_probe.py').read_text(encoding='utf8'))
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
old = 'assert all(int((split==i).sum())>200 for i in range(3))'
new = old + "\nsel=np.flatnonzero(split==2);fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel];assert len(fixed)>200"
assert inner.count(old) == 1
inner = inner.replace(old, new)
marker = "print('ALL_PHASE_UTILITY_SOURCE_CANDIDATES_FIXED_BEFORE_SOURCE_LABELS',flush=True)"
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + marker + r'''
import sys
payload=json.load(sys.stdin);n=len(fixed);features=[];baselines=[];candidates=[]
angles=core.tensor(np.stack((np.cos(phases),np.sin(phases)),1).astype(np.float32),'cpu')
for view in views:
 v=data[view];bf=v['base_feat'][:,None,:].expand(-1,12,-1);bp=v['base_prob'][:,None,:].expand(-1,12,-1);rp=v['raw_prob'][:,None,:].expand(-1,12,-1);en=v['energy'][:,None,:].expand(-1,12,-1);an=angles[None,:,:].expand(n,-1,-1)
 features.append(torch.cat((bf,v['phase_feat'],bp,rp,v['phase_prob'],en,an),2));baselines.append(v['base_prob'].argmax(1).cpu());candidates.append(v['phase_prob'].argmax(2).cpu())
features=torch.cat(features).reshape(-1,342);base=np.from_dlpack(torch.cat(baselines).contiguous());candidate=np.from_dlpack(torch.cat(candidates).contiguous())
mean=torch.tensor(payload['feature_mean'],dtype=torch.float32);std=torch.tensor(payload['feature_std'],dtype=torch.float32);weight=torch.tensor(payload['class_weights'],dtype=torch.float32)
model=nn.Sequential(nn.Linear(342,128),nn.GELU(),nn.Linear(128,3));model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in payload['model_state'].items()});model.eval();scores=[]
with torch.no_grad():
 for start in range(0,len(features),4096):
  scaled=((features[start:start+4096]-mean)/std).clamp(-10,10);prob=F.softmax(model(scaled)-weight.log(),1);scores.append((prob[:,1]-prob[:,2]).cpu())
scores=np.from_dlpack(torch.cat(scores).contiguous()).reshape(len(views)*n,12);disagree=candidate!=base[:,None];masked=scores.copy();masked[~disagree]=-np.inf;choice=masked.argmax(1);bestscore=masked[np.arange(len(masked)),choice];use=np.isfinite(bestscore)&(bestscore>payload['threshold']);out=base.copy();out[use]=candidate[np.arange(len(out))[use],choice[use]]
print('REPLAY_PHASE_CHOICES_FIXED_BEFORE_SOURCE_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1)
ref=np.from_dlpack(data['clean']['raw_prob']);teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target);per={}
for i,view in enumerate(views):
 sl=slice(i*n,(i+1)*n);old=base[sl]==target;new=out[sl]==target;lm=valid&labeled
 per[view]=dict(valid_count=int(valid.sum()),L_s_count=int(lm.sum()),U_confident_count=int((valid&~labeled).sum()),baseline=int((valid&old).sum()),student=int((valid&new).sum()),repairs=int((valid&~old&new).sum()),damages=int((valid&old&~new).sum()),L_s_baseline=int((lm&old).sum()),L_s_student=int((lm&new).sum()),switches=int((valid&use[sl]).sum()))
clean_net=int((out[:n]==teacher_target).sum()-(base[:n]==teacher_target).sum())
print('PHASE_UTILITY_REPLAY '+json.dumps(dict(heldout=per,heldout_clean_teacher_net=clean_net,source_ids=n,test_IQ_reads=0,predictions_before_source_labels=True),ensure_ascii=False),flush=True)
'''
payload = {k: result[k] for k in ('feature_mean', 'feature_std', 'class_weights', 'model_state')}
payload['threshold'] = result['selected']['threshold']
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,input=PAYLOAD,text=True,check=True,timeout=1800)
'''.replace('INNER', repr(inner), 1).replace('PAYLOAD', repr(json.dumps(payload, separators=(',', ':'))), 1)
proc = subprocess.run(['ssh', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=1820)
stdout = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(line for line in stdout.splitlines() if not line.startswith('PHASE_UTILITY_REPLAY ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
line = next(x[len('PHASE_UTILITY_REPLAY '):] for x in stdout.splitlines() if x.startswith('PHASE_UTILITY_REPLAY '))
replay = json.loads(line)
assert replay['heldout'] == result['heldout'] and replay['heldout_clean_teacher_net'] == result['heldout_clean_teacher_net']
assert replay['source_ids'] == result['heldout_ids'] and replay['predictions_before_source_labels'] and replay['test_IQ_reads'] == 0
dest = p / 'phase1_source_phase_utility_replay.json'
if dest.exists():
    assert json.loads(dest.read_text(encoding='utf8')) == replay
else:
    dest.write_text(json.dumps(replay, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print('VERIFIED independent fixed-model source phase-utility replay', dest)
