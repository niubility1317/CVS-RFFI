"""Assess a received/DURA-only noise proxy against source simulator SNR."""
import ast
import json
import subprocess
from pathlib import Path

p = Path(__file__).parent
tree = ast.parse((p / 'phase1_source_snr_diagnostic.py').read_text(encoding='utf8'))
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
old = "data[view]=dict(rawp=rawp,basep=basep,snr=np.asarray(snr),lock=np.asarray(lock),state=np.asarray(state),mse=aligned_mse(z,clean_complex));print('SNR_SOURCE_PREDICTIONS_FIXED',view,flush=True)"
new = "received_complex=received[:,0].astype(np.float64)+1j*received[:,1];raw_energy=np.mean(abs(received_complex[:,64:-64])**2,axis=1).clip(min=1e-20);residual_fraction=np.clip(wide_mse(z,received_complex)/raw_energy,1e-8,1.);proxy=10*np.log10(np.maximum(1-residual_fraction,1e-8)/residual_fraction);data[view]=dict(snr=np.asarray(snr),proxy=proxy,residual_fraction=residual_fraction);print('BLIND_SNR_SOURCE_PROXY_FIXED',view,flush=True)"
assert inner.count(old) == 1
inner = inner.replace(old, new)
old = 'for view in views:\n snr=[];lock=[];state=[]'
new = r'''def wide_mse(wave,truth):
 pad=np.pad(wave,((0,0),(64,64)),mode='reflect');target=truth[:,64:-64];best=np.full(len(wave),np.inf)
 for shift in range(-64,65):
  part=pad[:,64+shift:64+shift+256][:,64:-64];gain=np.sum(np.conj(part)*target,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-target)**2,axis=1);best=np.minimum(best,mse)
 return best
test_rng=np.random.default_rng(2026110459);test=test_rng.normal(size=(1,256))+1j*test_rng.normal(size=(1,256));pad=np.pad(test,((0,0),(64,64)),mode='reflect');shifted=complex(.4,.7)*pad[:,64+61:64+61+256];assert float(wide_mse(test,shifted)[0])<1e-20
''' + old
assert inner.count(old) == 1
inner = inner.replace(old, new)
marker = "print('ALL_SNR_SOURCE_PREDICTIONS_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)"
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + marker + r'''
def ranks(x):return np.argsort(np.argsort(x)).astype(np.float64)
results={};thresholds=(-10.,-5.,0.,5.,10.,15.,20.)
for view in views[1:]:
 v=data[view];true=v['snr'];proxy=v['proxy'];low=true<10.;groups=[]
 for t in thresholds:
  pred=proxy<t;groups.append(dict(threshold_db=t,true_low=int(low.sum()),selected=int(pred.sum()),true_positive=int((pred&low).sum()),false_positive=int((pred&~low).sum()),false_negative=int((~pred&low).sum()),true_negative=int((~pred&~low).sum())))
 results[view]=dict(count=n,simulated_snr_quantiles_db=np.quantile(true,[.05,.25,.5,.75,.95]).tolist(),blind_proxy_quantiles_db=np.quantile(proxy,[.05,.25,.5,.75,.95]).tolist(),spearman=float(np.corrcoef(ranks(true),ranks(proxy))[0,1]),gates=groups)
print('SOURCE_BLIND_SNR_PROXY '+json.dumps(dict(status='SOURCE_ONLY_BLIND_SNR_PROXY_WIDE',source_ids=n,receiver=8,day=3,clean_proxy_quantiles_db=np.quantile(data['clean']['proxy'],[.05,.25,.5,.75,.95]).tolist(),results=results,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,
 caveat='The proxy uses only received IQ and DURA, with a best-fit complex gain and shift within ±64 over the central 128 samples; no clean source or simulator SNR is used in the proxy. Simulated added-noise SNR is a privileged post-hoc reference, not total or real satellite SNR. This source receiver/day was observed in previous diagnostics; frozen models saw source records.'),ensure_ascii=False),flush=True)
'''
compile(inner, '<source-blind-snr-proxy>', 'exec')
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=900)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=920)
stdout = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(s for s in stdout.splitlines() if not s.startswith('SOURCE_BLIND_SNR_PROXY ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
line = next(s[len('SOURCE_BLIND_SNR_PROXY '):] for s in stdout.splitlines() if s.startswith('SOURCE_BLIND_SNR_PROXY '))
result = json.loads(line)
assert result['U_s_labels_read'] == result['V_reads'] == result['test_IQ_reads'] == 0
for view in result['results'].values():
    for q in view['gates']:
        assert q['true_positive'] + q['false_negative'] == q['true_low']
        assert q['true_positive'] + q['false_positive'] == q['selected']
        assert q['true_positive'] + q['false_positive'] + q['false_negative'] + q['true_negative'] == result['source_ids']
dest = p / 'phase1_source_blind_snr_proxy_wide.json'
if dest.exists():
    assert json.loads(dest.read_text(encoding='utf8')) == result
else:
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print('VERIFIED source received/DURA SNR proxy', dest)
