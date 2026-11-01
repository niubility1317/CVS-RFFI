"""Diagnose whether DURA follows or removes global input carrier phase."""
import ast
import json
import subprocess
from pathlib import Path

p = Path(__file__).parent
tree = ast.parse((p / 'phase1_source_equivariant_dura.py').read_text(encoding='utf8'))
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
changes = [
    ("anchors=[];rawp=[];basep=[];averages=[];disagreement=[]", "anchors=[];rawp=[];basep=[];averages=[];disagreement=[];orbits=[]"),
    ("stack=torch.stack(parts,1);assert torch.allclose", "stack=torch.stack(parts,1);orbits.append(torch.stack((stack.real,stack.imag),2).cpu());assert torch.allclose"),
    ("data[view]=dict(rawp=rawp,basep=basep,anchor=base,probs=np.stack(probs),candidates=np.stack(candidates),gate=gate,proxy=proxy,snr=np.asarray(snrs),variation=variation)", "data[view]=dict(rawp=rawp,basep=basep,anchor=base,probs=np.stack(probs),candidates=np.stack(candidates),gate=gate,proxy=proxy,snr=np.asarray(snrs),variation=variation,orbit=np.from_dlpack(torch.cat(orbits).contiguous()))"),
]
for old, new in changes:
    assert inner.count(old) == 1, old[:100]
    inner = inner.replace(old, new)
marker = "print('ALL_EQUIVARIANT_SOURCE_CANDIDATES_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)"
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + marker + r'''
results={}
for view in views:
 v=data[view];mask=(split==1)&v['gate'];anchor=v['anchor'][mask];den=np.mean(abs(anchor)**2,axis=1).clip(min=1e-20);phase_rows=[]
 if len(anchor):
  for i,theta in enumerate(phases):
   back=v['orbit'][mask,i,0].astype(np.float64)+1j*v['orbit'][mask,i,1];raw=back*np.exp(1j*theta);direct=np.mean(abs(raw-anchor)**2,axis=1)/den;equiv=np.mean(abs(raw-anchor*np.exp(1j*theta))**2,axis=1)/den;corr=np.sum(np.conj(anchor)*raw,axis=1)/np.sqrt(np.maximum(np.sum(abs(anchor)**2,axis=1)*np.sum(abs(raw)**2,axis=1),1e-20));phase_rows.append(dict(input_phase_deg=float(theta*180/np.pi),direct_nmse_median=float(np.median(direct)),equivariant_nmse_median=float(np.median(equiv)),normalized_complex_correlation_mean_real=float(np.mean(corr.real)),normalized_complex_correlation_mean_imag=float(np.mean(corr.imag))))
 results[view]=dict(gated_source_validation_count=int(mask.sum()),phase_responses=phase_rows)
print('SOURCE_DURA_PHASE_RESPONSE '+json.dumps(dict(status='SOURCE_ONLY_DURA_PHASE_RESPONSE',source_ids=n,validation_ids=int((split==1).sum()),results=results,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,
 caveat='Raw model outputs are compared with the zero-phase DURA output and with its input-phase-rotated version without fitting a complex gain. This source-only response test uses no transmitter labels; source clean IQ is used only to simulate received signals. Frozen models saw source records. Simulated satellite is not validated real satellite data.'),ensure_ascii=False),flush=True)
'''
compile(inner, '<source-dura-phase-response>', 'exec')
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=1800)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=1820)
stdout = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(s for s in stdout.splitlines() if not s.startswith('SOURCE_DURA_PHASE_RESPONSE ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
line = next(s[len('SOURCE_DURA_PHASE_RESPONSE '):] for s in stdout.splitlines() if s.startswith('SOURCE_DURA_PHASE_RESPONSE '))
result = json.loads(line)
assert result['U_s_labels_read'] == result['V_reads'] == result['test_IQ_reads'] == 0
dest = p / 'phase1_source_dura_phase_response.json'
if dest.exists():
    assert json.loads(dest.read_text(encoding='utf8')) == result
else:
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print('VERIFIED source DURA phase response', dest)
