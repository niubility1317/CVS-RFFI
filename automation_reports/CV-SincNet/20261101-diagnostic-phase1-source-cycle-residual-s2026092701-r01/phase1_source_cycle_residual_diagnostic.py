"""Test whether source received IQ contains clean residual absent from DURA."""
import ast
import json
import subprocess
from pathlib import Path

p = Path(__file__).parent
tree = ast.parse((p / 'phase1_source_cycle_projection_probe.py').read_text(encoding='utf8'))
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
assert inner.count('keep=split>0') == 1
inner = inner.replace('keep=split>0', 'keep=split==2')
assert inner.count('assert n>300') == 1
inner = inner.replace('assert n>300', 'assert n>200')
marker = 'for alpha in alphas:'
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + r'''
 data[view]=dict(anchor=z,received=received[:,0].astype(np.float64)+1j*received[:,1]);print('RESIDUAL_SOURCE_SIGNALS_FIXED',view,flush=True)
def aligned_to_reference(wave,reference):
 pad=np.pad(wave,((0,0),(16,16)),mode='reflect');crop=reference[:,16:-16];best=np.full(len(wave),np.inf);chosen=np.zeros_like(wave)
 for shift in range(-16,17):
  full=pad[:,16+shift:16+shift+256];part=full[:,16:-16];gain=np.sum(np.conj(part)*crop,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-crop)**2,axis=1);use=mse<best;best[use]=mse[use];chosen[use]=gain[use,None]*full[use]
 return chosen,best
synthetic_rng=np.random.default_rng(2026110419);synthetic=synthetic_rng.normal(size=(1,256))+1j*synthetic_rng.normal(size=(1,256));recovered,error_test=aligned_to_reference(synthetic*complex(2,.3),synthetic);assert float(error_test[0])<1e-20 and np.max(abs(recovered-synthetic))<1e-10
pc,_=project(clean_complex);qc=clean_complex-pc;den=np.sum(abs(qc)**2,axis=1).clip(min=1e-20);results={}
for view in views:
 v=data[view];a,a_mse=aligned_to_reference(v['anchor'],clean_complex);r,r_mse=aligned_to_reference(v['received'],clean_complex);pa,_=project(a);pr,_=project(r);qa=a-pa;qr=r-pr;delta=qr-qa;error=qc-qa;d2=np.sum(abs(delta)**2,axis=1);beta=np.clip(np.real(np.sum(np.conj(delta)*error,axis=1))/np.maximum(d2,1e-20),0,1);mixed=qa+beta[:,None]*delta;before=np.sum(abs(qa-qc)**2,axis=1);after=np.sum(abs(mixed-qc)**2,axis=1);corr=np.real(np.sum(np.conj(qa)*qc,axis=1))/np.sqrt(np.maximum(np.sum(abs(qa)**2,axis=1)*den,1e-20))
 results[view]=dict(source_ids=n,clean_residual_energy_fraction=float(np.sum(abs(qc)**2)/np.maximum(np.sum(abs(clean_complex)**2),1e-20)),dura_residual_relative_mse=float(before.sum()/den.sum()),optimistic_blend_residual_relative_mse=float(after.sum()/den.sum()),oracle_relative_improvement=float((before.sum()-after.sum())/max(before.sum(),1e-20)),samples_with_over_1pct_improvement=int((after<.99*before).sum()),mean_oracle_received_fraction=float(beta.mean()),dura_clean_residual_correlation_median=float(np.median(corr)),dura_aligned_iq_mse=float(a_mse.mean()),received_aligned_iq_mse=float(r_mse.mean()))
print('SOURCE_CYCLE_RESIDUAL '+json.dumps(dict(status='SOURCE_ONLY_CLEAN_ALIGNED_CYCLE_RESIDUAL_DIAGNOSTIC',source_ids=n,receiver=8,day=3,results=results,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,
 caveat='DURA and received IQ are separately aligned with paired clean IQ, and the blend coefficient is optimized per sample using the clean residual. Both are privileged source diagnostics, not deployable inference and not an upper bound on more general restorers. Clean source IQ was not available to the frozen DURA model; simulator is not validated real satellite data. Source IDs may overlap other prior probes.'),ensure_ascii=False),flush=True)
'''
compile(inner, '<source-cycle-residual>', 'exec')
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=600)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=620)
stdout = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(s for s in stdout.splitlines() if not s.startswith('SOURCE_CYCLE_RESIDUAL ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
line = next(s[len('SOURCE_CYCLE_RESIDUAL '):] for s in stdout.splitlines() if s.startswith('SOURCE_CYCLE_RESIDUAL '))
result = json.loads(line)
assert result['U_s_labels_read'] == result['V_reads'] == result['test_IQ_reads'] == 0 and result['receiver'] == 8
for v in result['results'].values():
    assert v['optimistic_blend_residual_relative_mse'] <= v['dura_residual_relative_mse'] + 1e-12
dest = p / 'phase1_source_cycle_residual_diagnostic.json'
if dest.exists():
    assert json.loads(dest.read_text(encoding='utf8')) == result
else:
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print('VERIFIED source cycle residual diagnostic', dest)
