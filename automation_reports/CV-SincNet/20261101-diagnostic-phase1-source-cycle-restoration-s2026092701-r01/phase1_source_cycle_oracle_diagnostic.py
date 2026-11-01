"""Source-only label-assisted headroom and paired-IQ oracle for cycle candidates."""
import ast
import json
import subprocess
from pathlib import Path

p = Path(__file__).parent
source = (p / 'phase1_source_cycle_projection_probe.py').read_text(encoding='utf8')
tree = ast.parse(source)
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
marker = "print('ALL_CYCLE_SOURCE_CANDIDATES_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)"
assert inner.count(marker) == 1
prefix = inner.split(marker, 1)[0] + marker
parsed = ast.parse(inner)
aligned = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'aligned_mse')
inner = prefix + '\n' + ast.get_source_segment(inner, aligned) + r'''
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=data['clean']['rawp'];teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target)
validation=[]
for i,alpha in enumerate(alphas):
 changes={}
 for view in views[1:]:
  v=data[view];mask=(split==1)&valid;base=v['basep'].argmax(1);out=v['probs'][i].argmax(1);changes[view]=int((mask&(out==target)).sum()-(mask&(base==target)).sum())
 validation.append(dict(candidate=i,alpha=alpha,per_view=changes,net=sum(changes.values())))
mask=split==2;valid_hold=valid[mask];l_hold=labeled[mask];truth=target[mask];results={}
for view in views:
 v=data[view];truth_iq=clean_complex[mask];base_pred=v['basep'][mask].argmax(1);candidate_pred=v['probs'][:,mask,:].argmax(2);base_err=aligned_mse(v['anchor'][mask],truth_iq);candidate_err=np.stack([aligned_mse(v['candidates'][i][mask],truth_iq) for i in range(len(alphas))]);base_correct=base_pred==truth;candidate_correct=candidate_pred==truth[None,:];class_oracle=base_correct|candidate_correct.any(0);fidelity_oracle=base_correct|np.any(candidate_correct&(candidate_err<base_err[None,:]),axis=0);best_error=np.minimum(base_err,candidate_err.min(0));per_alpha=[]
 for i,alpha in enumerate(alphas):
  correct=candidate_correct[i];per_alpha.append(dict(alpha=alpha,correct=int((valid_hold&correct).sum()),L_s_correct=int((valid_hold&l_hold&correct).sum()),aligned_mse=float(candidate_err[i].mean()),quality_improved=int((candidate_err[i]<base_err).sum()),quality_worsened=int((candidate_err[i]>base_err).sum())))
 results[view]=dict(valid_count=int(valid_hold.sum()),L_s_count=int((valid_hold&l_hold).sum()),baseline_correct=int((valid_hold&base_correct).sum()),L_s_baseline_correct=int((valid_hold&l_hold&base_correct).sum()),class_oracle_correct=int((valid_hold&class_oracle).sum()),L_s_class_oracle_correct=int((valid_hold&l_hold&class_oracle).sum()),fidelity_class_oracle_correct=int((valid_hold&fidelity_oracle).sum()),L_s_fidelity_class_oracle_correct=int((valid_hold&l_hold&fidelity_oracle).sum()),aligned_mse_baseline=float(base_err.mean()),aligned_mse_oracle=float(best_error.mean()),quality_oracle_improved=int((best_error<base_err).sum()),all_count=int(mask.sum()),per_alpha=per_alpha)
print('SOURCE_CYCLE_ORACLE '+json.dumps(dict(status='SOURCE_ONLY_LABEL_ASSISTED_CYCLE_ORACLE',source_ids=n,validation_ids=int((split==1).sum()),heldout_ids=int(mask.sum()),validation=validation,results=results,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,candidates_before_source_tx_labels=True,
 caveat='Oracles use paired source clean IQ and/or mixed source targets per record and cannot run at inference. They are bounds only for these seven fixed waveforms and this source sample, not for new restoration networks or target performance. The 20-sample period was selected after earlier source inspection; fixed experts saw source data and simulator is not validated satellite data.'),ensure_ascii=False),flush=True)
'''
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=1800)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=1820)
stdout = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(s for s in stdout.splitlines() if not s.startswith('SOURCE_CYCLE_ORACLE ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
line = next(s[len('SOURCE_CYCLE_ORACLE '):] for s in stdout.splitlines() if s.startswith('SOURCE_CYCLE_ORACLE '))
result = json.loads(line)
baseline = json.loads((p / 'phase1_source_cycle_projection_probe.json').read_text(encoding='utf8'))
assert (result['source_ids'], result['validation_ids'], result['heldout_ids']) == (baseline['source_ids'], baseline['validation_ids'], baseline['heldout_ids'])
for actual, expected in zip(result['validation'], baseline['validation']):
    assert all(actual[k] == expected[k] for k in ('candidate', 'alpha', 'per_view', 'net'))
assert result['U_s_labels_read'] == result['V_reads'] == result['test_IQ_reads'] == 0 and result['candidates_before_source_tx_labels']
dest = p / 'phase1_source_cycle_oracle_diagnostic.json'
if dest.exists():
    assert json.loads(dest.read_text(encoding='utf8')) == result
else:
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print('VERIFIED source cycle validation replay and label-assisted oracle', dest)
