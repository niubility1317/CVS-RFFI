"""Inspect unlabeled source temporal correlation without assuming a preamble."""
import json
import subprocess
from pathlib import Path

inner = r'''
from pathlib import Path
import json,numpy as np,torch
from experiments.pamf_phase1_flow import experiment as core,physical_rich_router_experiment as g
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
models=g.load_models(root,'cpu');teacher,contract,dura,*_=models;cache,clean,roles=core.data(root,contract)
from leo_practical.channel import Config,receiver_for_session,stable_seed,normalize_input
from leo_practical.residual import ResidualChannel
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);rx=np.load(cache/'train/rx.npy',allow_pickle=False);day=np.load(cache/'train/day.npy',allow_pickle=False);rng=np.random.default_rng(2026110401)
available=np.flatnonzero((roles=='U_s')&np.isin(rx,[4,6,8])&(day==3));assert len(available)>=500;fixed=np.sort(rng.choice(available,size=min(2048,len(available)),replace=False));n=len(fixed);assert len(np.unique(fixed))==n
views=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban');clean_iq=np.asarray(clean[fixed],dtype=np.float32);lags=np.arange(1,129);all_records={}
def correlation(iq):
 z=iq[:,0,:].astype(np.float64)+1j*iq[:,1,:];out=[]
 for lag in lags:
  x=z[:,:-lag];y=z[:,lag:];den=np.sqrt(np.sum(abs(x)**2,axis=1)*np.sum(abs(y)**2,axis=1));out.append(np.sum(np.conj(x)*y,axis=1)/np.maximum(den,1e-20))
 return np.stack(out,1)
reference=correlation(clean_iq)
for view in views:
 if view=='clean':received=clean_iq
 else:
  cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=view,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');received=[]
  for j in fixed:
   pid=str(ids[j]);session='rx%d:day%d:eq1'%(int(rx[j]),int(day[j]));hw=receiver_for_session(cfg,2026110403,session);v=np.asarray(clean[j],dtype=np.float64);signal,_=normalize_input(v[0]+1j*v[1]);result,meta=ResidualChannel(cfg,stable_seed(2026110402,'source-cyclostationary-2026110402',view,pid),hw).process(signal)
   assert np.isfinite(result).all() and not meta['channel_equalization_applied'];received.append(np.stack((result.real,result.imag)).astype(np.float32))
  received=np.stack(received)
 anchors=[]
 with torch.no_grad():
  for start in range(0,n,64):anchors.append(dura(core.tensor(received[start:start+64],'cpu')).cpu())
 anchor=np.from_dlpack(torch.cat(anchors).contiguous());observed=correlation(received);restored=correlation(anchor)
 all_records[view]=dict(observed=observed,restored=restored)
 print('SOURCE_CORRELATION_VIEW_FIXED',view,flush=True)
def stats(a):return dict(median_abs=float(np.median(abs(a))),coherence=float(abs(np.mean(a))/max(np.mean(abs(a)),1e-20)))
top=[]
for lag in np.argsort(np.median(abs(reference[:,7:]),axis=0))[::-1][:12]+8:top.append(dict(lag=int(lag),**stats(reference[:,lag-1])))
selected=sorted(set([1,2,4,8,16,20,32,40,64,80,100,128]+[x['lag'] for x in top]));results={}
for view in views:
 observed=all_records[view]['observed'];restored=all_records[view]['restored'];rows=[]
 for lag in selected:
  j=lag-1;rows.append(dict(lag=int(lag),clean=stats(reference[:,j]),input=stats(observed[:,j]),dura=stats(restored[:,j]),input_error=float(np.mean(abs(observed[:,j]-reference[:,j]))),dura_error=float(np.mean(abs(restored[:,j]-reference[:,j])))))
 results[view]=rows
print('SOURCE_CYCLOSTATIONARY '+json.dumps(dict(status='SOURCE_ONLY_UNLABELED_CORRELATION_PROBE',source_ids=n,roles=['U_s'],receivers=[4,6,8],day=3,top_clean_lags_at_least_8=top,results=results,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,caveat='The 256-sample equalized window is not assumed to be a known or aligned preamble. Normalized sample autocorrelation measures repeatable second-order structure, not transmitter identity or recovered information. Clean source IQ is used for paired source diagnostics, never inference; source IDs may overlap previous probes. Satellite simulation is not a validated real link.'),ensure_ascii=False),flush=True)
'''
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=1800)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.Popen(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf8', errors='replace', bufsize=1)
proc.stdin.write(remote)
proc.stdin.close()
marker = 'SOURCE_CYCLOSTATIONARY '
payload = None
for line in proc.stdout:
    if line.startswith(marker):
        payload = line[len(marker):]
    else:
        print(line, end='', flush=True)
rc = proc.wait()
if rc == 0:
    if payload is None:
        raise ValueError('source correlation result marker missing')
    dest = Path(__file__).parent / 'phase1_source_cyclostationary_probe.json'
    if dest.exists():
        assert json.loads(dest.read_text(encoding='utf8')) == json.loads(payload)
    else:
        dest.write_text(payload, encoding='utf8', newline='\n')
    print('SAVED_SOURCE_CORRELATION_RESULT', dest, flush=True)
raise SystemExit(rc)
