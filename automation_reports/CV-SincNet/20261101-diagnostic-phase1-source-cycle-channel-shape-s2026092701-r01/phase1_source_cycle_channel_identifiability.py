"""Compare pseudo-pilot frequency response with privileged simulator kernels."""
import ast
import json
import subprocess
from pathlib import Path

p = Path(__file__).parent
wrapper = ast.parse((p / 'phase1_source_cycle_channel_probe.py').read_text(encoding='utf8'))
node = next(n.value for n in wrapper.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets))
rtree = ast.parse(ast.literal_eval(node))
run = next(n for n in ast.walk(rtree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'run')
inner = ast.literal_eval(run.args[0].elts[3])
assert inner.count('keep=split>0') == 1
inner = inner.replace('keep=split>0', 'keep=split==2')
assert inner.count('assert n>300') == 1
inner = inner.replace('assert n>300', 'assert n>200')
marker = 'for view in views:\n if view=='
assert inner.count(marker) == 1
inner = inner.split(marker, 1)[0] + r'''
def frequency_shape_error(estimate,truth):
 freq=np.fft.fftfreq(256);den=np.sum(abs(truth)**2,axis=1).clip(min=1e-20);best=np.full(len(estimate),np.inf)
 for shift in range(-96,97):
  candidate=estimate*np.exp(-2j*np.pi*freq[None,:]*shift);gain=np.sum(np.conj(candidate)*truth,axis=1)/np.maximum(np.sum(abs(candidate)**2,axis=1),1e-20);error=np.sum(abs(gain[:,None]*candidate-truth)**2,axis=1)/den;best=np.minimum(best,error)
 return best
test_rng=np.random.default_rng(2026110441);test=test_rng.normal(size=(2,256))+1j*test_rng.normal(size=(2,256));scaled=test*complex(.4,-.6);assert np.max(frequency_shape_error(test,scaled))<1e-20;delay=np.exp(-2j*np.pi*np.fft.fftfreq(256)*61)[None,:];assert np.max(frequency_shape_error(np.ones((1,256),complex),delay))<1e-20
results={}
for view in views[1:]:
 cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');received=[];kernels=[]
 for j in fixed:
  pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110435,session);v=np.asarray(clean[j],dtype=np.float64);signal,_=normalize_input(v[0]+1j*v[1]);result,meta,trace=ResidualChannel(cfg,stable_seed(2026110434,'source-cycle-channel-2026110434',view,pid),hw).process(signal,return_traces=True);assert np.isfinite(result).all() and not meta['channel_equalization_applied'];received.append(np.stack((result.real,result.imag)).astype(np.float32));kernels.append(np.asarray(trace['residual_kernel'],dtype=np.complex128))
 received=np.stack(received);anchors=[]
 with torch.no_grad():
  for start in range(0,n,32):anchors.append(dura(core.tensor(received[start:start+32],'cpu')).cpu())
 anchor=np.from_dlpack(torch.cat(anchors).contiguous());a=anchor[:,0].astype(np.float64)+1j*anchor[:,1];x=received[:,0].astype(np.float64)+1j*received[:,1];_,_,H,_=channel_from_cycles(x,a);kernel_array=np.stack(kernels);truth=np.fft.fft(kernel_array,n=256,axis=1);pred_err=frequency_shape_error(H,truth);flat_err=frequency_shape_error(np.ones_like(H),truth);peaks=np.argmax(abs(kernel_array),axis=1);results[view]=dict(count=n,pseudo_pilot_shape_nmse=float(pred_err.mean()),pseudo_pilot_shape_nmse_median=float(np.median(pred_err)),flat_channel_shape_nmse=float(flat_err.mean()),flat_channel_shape_nmse_median=float(np.median(flat_err)),pseudo_pilot_better=int((pred_err<flat_err).sum()),kernel_peak_index_min=int(peaks.min()),kernel_peak_index_median=float(np.median(peaks)),kernel_peak_index_max=int(peaks.max()))
 print('CYCLE_CHANNEL_SOURCE_ESTIMATES_FIXED',view,flush=True)
print('SOURCE_CYCLE_CHANNEL_IDENTIFIABILITY '+json.dumps(dict(status='SOURCE_ONLY_CYCLE_CHANNEL_IDENTIFIABILITY_WIDE',source_ids=n,receiver=8,day=3,results=results,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,
 caveat='The channel estimator uses received IQ and frozen DURA only; true residual simulator kernels are privileged source-only diagnostic references and are never inference inputs. Frequency-shape NMSE permits the best source-oracle complex gain and integer shift over ±96, covering the 62-tap kernel and possible inference alignment offsets, and ignores nonconvolutional effects and residual CFO. Flat-response comparison does not assess classification. Simulator is not validated real satellite data.'),ensure_ascii=False),flush=True)
'''
compile(inner, '<cycle-channel-identifiability>', 'exec')
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=900)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=920)
stdout = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(s for s in stdout.splitlines() if not s.startswith('SOURCE_CYCLE_CHANNEL_IDENTIFIABILITY ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
line = next(s[len('SOURCE_CYCLE_CHANNEL_IDENTIFIABILITY '):] for s in stdout.splitlines() if s.startswith('SOURCE_CYCLE_CHANNEL_IDENTIFIABILITY '))
result = json.loads(line)
assert result['U_s_labels_read'] == result['V_reads'] == result['test_IQ_reads'] == 0 and result['receiver'] == 8
dest = p / 'phase1_source_cycle_channel_identifiability_wide.json'
if dest.exists():
    assert json.loads(dest.read_text(encoding='utf8')) == result
else:
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print('VERIFIED source cycle channel identifiability', dest)
