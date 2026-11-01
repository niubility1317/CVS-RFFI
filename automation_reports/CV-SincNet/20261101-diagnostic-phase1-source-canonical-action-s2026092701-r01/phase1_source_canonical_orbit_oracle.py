"""Label-assisted source headroom for phase-rotated-input DURA waveforms."""
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
for view in views:
 v=data[view];old_orbit=v['orbit'];canonical=[]
 for phase in range(8):
  z=old_orbit[:,phase,0].astype(np.float64)+1j*old_orbit[:,phase,1];z=z*np.exp(1j*phases[phase]);canonical.append(np.stack((z.real,z.imag),1).astype(np.float32))
 v['orbit']=np.stack(canonical,1);pred=[]
 with torch.no_grad():
  for phase in range(8):
   out=[]
   for start in range(0,n,64):
    x=core.tensor(v['orbit'][start:start+64,phase],'cpu');out.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu())
   pred.append(np.from_dlpack(torch.cat(out).contiguous()).argmax(1))
 v['orbit_prediction']=np.stack(pred);assert np.array_equal(v['orbit_prediction'][0],v['basep'].argmax(1));print('DURA_ORBIT_SOURCE_PREDICTIONS_FIXED',view,flush=True)
print('ALL_DURA_ORBIT_PREDICTIONS_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)
labels=fs.source_labels(root,cache,clean,roles,contract);labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1);ref=data['clean']['rawp'];teacher_target=ref.argmax(1);valid=labeled|(ref.max(1)>=.95);target=np.where(labeled,labels[fixed],teacher_target)
def aligned_mse(wave,truth):
 pad=np.pad(wave,((0,0),(16,16)),mode='reflect');truth=truth[:,16:-16];best=np.full(len(wave),np.inf)
 for shift in range(-16,17):
  part=pad[:,16+shift:16+shift+256][:,16:-16];gain=np.sum(np.conj(part)*truth,axis=1)/np.maximum(np.sum(abs(part)**2,axis=1),1e-20);mse=np.mean(abs(gain[:,None]*part-truth)**2,axis=1);best=np.minimum(best,mse)
 return best
results={};validation=[]
for view in views:
 v=data[view];base=v['basep'].argmax(1);orbit=v['orbit_prediction'];valid_orbit=np.broadcast_to(v['gate'][None,:],orbit.shape).copy();valid_orbit[0]=True;correct=orbit==target[None,:];basecorrect=base==target;phase_ok=basecorrect|np.any(correct&valid_orbit,axis=0);wave=v['orbit'];quality=np.stack([aligned_mse(wave[:,i,0].astype(np.float64)+1j*wave[:,i,1],clean_complex) for i in range(8)]);fidelity=basecorrect|np.any(correct&valid_orbit&(quality<quality[0:1]),axis=0);per={}
 for part in (1,2):
  m=(split==part)&valid;lm=m&labeled;per[str(part)]=dict(count=int((split==part).sum()),valid_count=int(m.sum()),L_s_count=int(lm.sum()),baseline_correct=int((m&basecorrect).sum()),L_s_baseline_correct=int((lm&basecorrect).sum()),orbit_oracle_correct=int((m&phase_ok).sum()),L_s_orbit_oracle_correct=int((lm&phase_ok).sum()),fidelity_orbit_oracle_correct=int((m&fidelity).sum()),L_s_fidelity_orbit_oracle_correct=int((lm&fidelity).sum()),gated_count=int(((split==part)&v['gate']).sum()),phase_oracle_mean_mse=float(np.minimum(quality[0],np.where(valid_orbit,quality,np.inf).min(0))[split==part].mean()),baseline_mean_mse=float(quality[0,split==part].mean()))
 results[view]=per
for i,alpha in enumerate(alphas):
 changes={}
 for view in views[1:]:
  v=data[view];m=(split==1)&valid;base=v['basep'].argmax(1);out=v['probs'][i].argmax(1);changes[view]=int((m&(out==target)).sum()-(m&(base==target)).sum())
 validation.append(dict(candidate=i,alpha=alpha,net=sum(changes.values()),per_view=changes))
print('SOURCE_DURA_ORBIT_ORACLE '+json.dumps(dict(status='SOURCE_ONLY_LABEL_ASSISTED_CANONICAL_ORBIT_ORACLE',source_ids=n,validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),results=results,validation=validation,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,predictions_before_source_tx_labels=True,
 caveat='Each of eight waveforms is DURA applied to a globally rotated received signal, kept in its observed common output frame without inverse rotation. Oracle selection uses source targets and optionally paired clean IQ, so it cannot run at inference and bounds only these fixed candidates. Source sample and validation were previously inspected; frozen experts saw source records. Simulated satellite is not validated real satellite data.'),ensure_ascii=False),flush=True)
'''
compile(inner, '<source-dura-orbit-oracle>', 'exec')
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=1800)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=1820)
stdout = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(s for s in stdout.splitlines() if not s.startswith('SOURCE_DURA_ORBIT_ORACLE ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
line = next(s[len('SOURCE_DURA_ORBIT_ORACLE '):] for s in stdout.splitlines() if s.startswith('SOURCE_DURA_ORBIT_ORACLE '))
result = json.loads(line)
baseline = json.loads((p / 'phase1_source_equivariant_dura.json').read_text(encoding='utf8'))
for actual, expected in zip(result['validation'], baseline['validation']):
    assert all(actual[k] == expected[k] for k in ('candidate', 'alpha', 'net', 'per_view'))
assert result['U_s_labels_read'] == result['V_reads'] == result['test_IQ_reads'] == 0 and result['predictions_before_source_tx_labels']
dest = p / 'phase1_source_canonical_orbit_oracle.json'
if dest.exists():
    assert json.loads(dest.read_text(encoding='utf8')) == result
else:
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print('VERIFIED source canonical orbit validation and label-assisted oracle', dest)
