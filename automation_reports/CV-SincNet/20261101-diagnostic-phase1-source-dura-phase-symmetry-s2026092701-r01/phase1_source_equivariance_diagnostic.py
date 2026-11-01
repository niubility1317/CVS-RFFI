"""Diagnose phase-orbit waveform and classification degradation on source validation."""
import ast
import json
import subprocess
from pathlib import Path

p = Path(__file__).parent
tree = ast.parse((p / 'phase1_source_equivariant_dura.py').read_text(encoding='utf8'))
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
marker = "print('ALL_EQUIVARIANT_SOURCE_CANDIDATES_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)"
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + marker + r'''
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=data['clean']['rawp'];teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target)
def aligned_mse(wave,truth):
 pad=np.pad(wave,((0,0),(16,16)),mode='reflect');truth=truth[:,16:-16];best=np.full(len(wave),np.inf)
 for shift in range(-16,17):
  part=pad[:,16+shift:16+shift+256][:,16:-16];gain=np.sum(np.conj(part)*truth,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-truth)**2,axis=1);best=np.minimum(best,mse)
 return best
results={};validation=[]
for view in views:
 v=data[view];all_val=split==1;active=all_val&v['gate'];mask=all_val&valid;lm=mask&labeled;base=v['basep'].argmax(1);old=base==target;rows=[]
 if active.any():
  before=aligned_mse(v['anchor'][active],clean_complex[active]);base_rms=np.sqrt(np.mean(abs(v['anchor'][active])**2,axis=1)).clip(min=1e-12);variation=np.quantile(v['variation'][active],[.1,.5,.9]).tolist()
 else:before=np.array([]);base_rms=np.array([]);variation=None
 for i,alpha in enumerate(alphas):
  out=v['probs'][i].argmax(1);new=out==target
  if active.any():
   wave=v['candidates'][i][active];after=aligned_mse(wave,clean_complex[active]);energy=np.sqrt(np.mean(abs(wave)**2,axis=1))/base_rms;diff=np.sqrt(np.mean(abs(wave-v['anchor'][active])**2,axis=1))/base_rms;quality=dict(aligned_mse_before=float(before.mean()),aligned_mse_after=float(after.mean()),improved=int((after<before).sum()),worsened=int((after>before).sum()),median_output_rms_ratio=float(np.median(energy)),median_relative_change=float(np.median(diff)))
  else:quality=None
  rows.append(dict(alpha=alpha,valid_count=int(mask.sum()),baseline=int((mask&old).sum()),candidate=int((mask&new).sum()),repairs=int((mask&~old&new).sum()),damages=int((mask&old&~new).sum()),L_s_count=int(lm.sum()),L_s_baseline=int((lm&old).sum()),L_s_candidate=int((lm&new).sum()),quality_on_gate=quality))
 results[view]=dict(gated_count=int(active.sum()),orbit_variation_quantiles=variation,candidates=rows)
for i,alpha in enumerate(alphas):
 changes={v:results[v]['candidates'][i]['candidate']-results[v]['candidates'][i]['baseline'] for v in views[1:]};validation.append(dict(candidate=i,alpha=alpha,net=sum(changes.values()),per_view=changes))
print('SOURCE_EQUIVARIANCE_DIAGNOSTIC '+json.dumps(dict(status='SOURCE_ONLY_EQUIVARIANCE_DIAGNOSTIC',source_ids=n,validation_ids=int((split==1).sum()),results=results,validation=validation,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,candidates_before_source_tx_labels=True,
 caveat='The 8-way phase-orbit average uses received IQ and DURA only; source clean IQ and mixed source targets are used solely for post-hoc diagnostics. IQ MSE allows source-only ±16 shift and complex gain. Orbit variation measures disagreement among inverse-rotated DURA outputs, not necessarily physical restoration error. Source validation was previously inspected; experts saw source records and simulator is not validated real satellite data.'),ensure_ascii=False),flush=True)
'''
compile(inner, '<source-equivariance-diagnostic>', 'exec')
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=1800)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=1820)
stdout = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(s for s in stdout.splitlines() if not s.startswith('SOURCE_EQUIVARIANCE_DIAGNOSTIC ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
line = next(s[len('SOURCE_EQUIVARIANCE_DIAGNOSTIC '):] for s in stdout.splitlines() if s.startswith('SOURCE_EQUIVARIANCE_DIAGNOSTIC '))
result = json.loads(line)
baseline = json.loads((p / 'phase1_source_equivariant_dura.json').read_text(encoding='utf8'))
assert (result['source_ids'], result['validation_ids']) == (baseline['source_ids'], baseline['validation_ids'])
for actual, expected in zip(result['validation'], baseline['validation']):
    assert all(actual[k] == expected[k] for k in ('candidate', 'alpha', 'net', 'per_view'))
assert result['U_s_labels_read'] == result['V_reads'] == result['test_IQ_reads'] == 0 and result['candidates_before_source_tx_labels']
dest = p / 'phase1_source_equivariance_diagnostic.json'
if dest.exists():
    assert json.loads(dest.read_text(encoding='utf8')) == result
else:
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print('VERIFIED source equivariance validation replay and diagnostics', dest)
