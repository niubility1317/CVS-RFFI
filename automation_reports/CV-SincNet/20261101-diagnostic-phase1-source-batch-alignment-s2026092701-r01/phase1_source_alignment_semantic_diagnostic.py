"""Check whether paired clean alignment preserves frozen-base classification."""
import ast
import json
import subprocess
from pathlib import Path

p = Path(__file__).parent
wrapper = ast.parse((p / 'phase1_source_cycle_feature_net.py').read_text(encoding='utf8'))
node = next(n.value for n in wrapper.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets))
remote_old = ast.literal_eval(node)
rtree = ast.parse(remote_old)
run = next(n for n in ast.walk(rtree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'run')
inner = ast.literal_eval(run.args[0].elts[3])
old = 'clean_wave=np.stack((clean_aligned.real,clean_aligned.imag),1).astype(np.float32);target_features=[]'
new = 'clean_wave=np.stack((clean_aligned.real,clean_aligned.imag),1).astype(np.float32);target_features=[];target_predictions=[]'
assert inner.count(old) == 1
inner = inner.replace(old, new)
old = "for start in range(0,n,64):target_features.append(F.normalize(teacher.features(core.tensor(clean_wave[start:start+64],'cpu')),dim=1).cpu())"
new = "for start in range(0,n,64):\n   f=teacher.features(core.tensor(clean_wave[start:start+64],'cpu'));target_features.append(F.normalize(f,dim=1).cpu());target_predictions.append(F.softmax(teacher.classify_features(f),1).cpu())"
assert inner.count(old) == 1
inner = inner.replace(old, new)
old = "target_feature=torch.cat(target_features).contiguous(),basep=basep"
new = "target_feature=torch.cat(target_features).contiguous(),target_prediction=np.from_dlpack(torch.cat(target_predictions).contiguous()),basep=basep"
assert inner.count(old) == 1
inner = inner.replace(old, new)
marker = 'class CycleNet(torch.nn.Module):'
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + r'''
print('ALL_ALIGNED_CLEAN_PREDICTIONS_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);reference=data['clean']['rawp'];teacher_target=reference.argmax(1);valid=labeled|(reference.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target);results={}
for part in (0,1,2):
 mask=(split==part)&valid;lm=mask&labeled;views_result={}
 for view in views:
  q=data[view]['target_prediction'].argmax(1);views_result[view]=dict(valid_count=int(mask.sum()),L_s_count=int(lm.sum()),reference_correct=int((mask&(teacher_target==target)).sum()),aligned_correct=int((mask&(q==target)).sum()),aligned_agreement_with_reference=int((mask&(q==teacher_target)).sum()),L_s_reference_correct=int((lm&(teacher_target==target)).sum()),L_s_aligned_correct=int((lm&(q==target)).sum()))
 results[str(part)]=views_result
print('SOURCE_ALIGNMENT_SEMANTICS '+json.dumps(dict(status='SOURCE_ONLY_ALIGNED_CLEAN_SEMANTICS',source_ids=n,train_ids=int((split==0).sum()),validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),results=results,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,predictions_before_source_tx_labels=True,
 caveat='Clean-to-DURA alignment uses the paired source waveform only to construct a training target, never as restoration inference. This measures whether that privileged target preserves frozen-base labels. Only the new model receiver/day is held out; F and DURA saw source data. Source IDs were used in previous diagnostics and simulated satellite data is not a validated real link.'),ensure_ascii=False),flush=True)
'''
compile(inner, '<alignment-semantics>', 'exec')
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=1800)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=1820)
stdout = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(s for s in stdout.splitlines() if not s.startswith('SOURCE_ALIGNMENT_SEMANTICS ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
line = next(s[len('SOURCE_ALIGNMENT_SEMANTICS '):] for s in stdout.splitlines() if s.startswith('SOURCE_ALIGNMENT_SEMANTICS '))
result = json.loads(line)
assert result['U_s_labels_read'] == result['V_reads'] == result['test_IQ_reads'] == 0 and result['predictions_before_source_tx_labels']
dest = p / 'phase1_source_alignment_semantic_diagnostic.json'
if dest.exists():
    assert json.loads(dest.read_text(encoding='utf8')) == result
else:
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print('VERIFIED source aligned-clean semantic diagnostic', dest)
