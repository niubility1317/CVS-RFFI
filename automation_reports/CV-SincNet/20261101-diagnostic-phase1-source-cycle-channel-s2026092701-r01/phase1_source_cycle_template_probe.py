"""Test whether unlabeled phase-aligned source cycle templates generalize."""
import json
import subprocess
from pathlib import Path

inner = r'''
from pathlib import Path
import json,numpy as np,torch
from experiments.pamf_phase1_flow import experiment as core,physical_rich_router_experiment as g
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2);models=g.load_models(root,'cpu');teacher,contract,*_=models;cache,clean,roles=core.data(root,contract)
rx=np.load(cache/'train/rx.npy',allow_pickle=False);day=np.load(cache/'train/day.npy',allow_pickle=False);rng=np.random.default_rng(2026110431);train_pool=np.flatnonzero((roles=='U_s')&np.isin(rx,[1,3,4])&np.isin(day,[1,2]));val_pool=np.flatnonzero((roles=='U_s')&np.isin(rx,[4,6])&(day==3));hold_pool=np.flatnonzero((roles=='U_s')&(rx==8)&(day==3));assert min(len(train_pool),len(val_pool),len(hold_pool))>=500
train=np.sort(rng.choice(train_pool,size=min(2048,len(train_pool)),replace=False));val=np.sort(rng.choice(val_pool,size=min(1024,len(val_pool)),replace=False));held=np.sort(rng.choice(hold_pool,size=min(1024,len(hold_pool)),replace=False));fixed=np.concatenate((train,val,held));assert len(np.unique(fixed))==len(fixed);part=np.concatenate((np.zeros(len(train),int),np.ones(len(val),int),np.full(len(held),2,int)))
wave=np.asarray(clean[fixed],dtype=np.float64);z=wave[:,0,:]+1j*wave[:,1,:];period=20;cycles=np.arange(256)//period;lagcorr=np.sum(np.conj(z[:,:-period])*z[:,period:],axis=1);phi=np.angle(lagcorr);demod=z*np.exp(-1j*phi[:,None]*cycles);templates=np.stack([np.mean(demod[:,r::period],axis=1) for r in range(period)],1);power=np.sqrt(np.sum(abs(templates)**2,axis=1)).clip(min=1e-12);templates=templates/power[:,None]
def align(templates,reference):
 shifted=np.stack([np.roll(templates,s,axis=1) for s in range(period)],1);corr=np.sum(np.conj(shifted)*reference[None,None,:],axis=2);index=np.abs(corr).argmax(1);best=corr[np.arange(len(templates)),index];chosen=shifted[np.arange(len(templates)),index];phase=best/np.maximum(abs(best),1e-20);return chosen*phase[:,None],abs(best),index
test_rng=np.random.default_rng(2026110433);synthetic=test_rng.normal(size=20)+1j*test_rng.normal(size=20);synthetic/=np.linalg.norm(synthetic);recovered,corr_test,_=align((np.roll(synthetic,7)*np.exp(.43j))[None,:],synthetic);assert np.max(abs(recovered[0]-synthetic))<1e-10 and abs(float(corr_test[0])-1)<1e-10
ref=templates[0].copy();history=[]
for _ in range(5):
 aligned,corr,_=align(templates[part==0],ref);ref=aligned.mean(0);ref/=max(np.linalg.norm(ref),1e-12);history.append(float(np.median(corr)))
results={}
for group in (0,1,2):
 aligned,corr,shift=align(templates[part==group],ref);ft=np.fft.fft(aligned,axis=1);mean=ft.mean(0);rms=np.sqrt(np.mean(abs(ft)**2,axis=0));coherence=abs(mean)/np.maximum(rms,1e-20);results[str(group)]=dict(count=int((part==group).sum()),median_correlation=float(np.median(corr)),mean_correlation=float(np.mean(corr)),p10_correlation=float(np.quantile(corr,.1)),fraction_above_09=float(np.mean(corr>.9)),fraction_above_095=float(np.mean(corr>.95)),harmonic_coherence=coherence.tolist(),harmonic_median_amplitude=np.median(abs(ft),axis=0).tolist())
print('SOURCE_CYCLE_TEMPLATE '+json.dumps(dict(status='SOURCE_ONLY_UNLABELED_CYCLE_TEMPLATE',period=period,train_ids=len(train),validation_ids=len(val),heldout_ids=len(held),train_receivers=[1,3,4],train_days=[1,2],validation_receivers=[4,6],heldout_receiver=8,validation_heldout_day=3,fit_history=history,results=results,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,
 caveat='Template fitting uses only source U_s clean IQ; labels are not read. Each waveform is allowed a best circular shift and complex phase to align its 20-sample template, an oracle alignment for this diagnostic, not a deployment procedure. High similarity alone does not establish a known pilot or satellite channel recovery. Frozen F saw source records, and source IDs may overlap prior probes.'),ensure_ascii=False),flush=True)
'''
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
assert (release/'release_commit.txt').read_text().strip()=='b7beba9e6419a1ec4bf39c83cc17e2ee6478d5f0'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,check=True,timeout=600)
'''.replace('INNER', repr(inner), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', '-o', 'BatchMode=yes', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=620)
stdout = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(s for s in stdout.splitlines() if not s.startswith('SOURCE_CYCLE_TEMPLATE ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
line = next(s[len('SOURCE_CYCLE_TEMPLATE '):] for s in stdout.splitlines() if s.startswith('SOURCE_CYCLE_TEMPLATE '))
result = json.loads(line)
assert result['U_s_labels_read'] == result['V_reads'] == result['test_IQ_reads'] == 0
dest = Path(__file__).parent / 'phase1_source_cycle_template_probe.json'
if dest.exists():
    assert json.loads(dest.read_text(encoding='utf8')) == result
else:
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
print('VERIFIED source cycle template diagnostic', dest)
