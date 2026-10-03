"""Independent NumPy FIR/STFT/Gram recount and all-public-sample reporting."""
import argparse,csv,gzip,json,statistics
from pathlib import Path
import numpy as np
RUN='20261003-diagnostic-cvs-mirror-history-public-m8-r01'
DELAYS=(2,8,16);GROUPS=('noise','tone','periodic','dc','zero')
OUTPUTS=('packets.csv','frequency_pairs.csv.gz','frames.csv','embeddings.csv','summary.csv','summary.json','independent_recount.json')

def validate_completion(done):
    expected={(v,s):f'{v}-s{s}' for v in ('mirror_energy','mirror_subspace') for s in range(2026092701,2026092705)}
    rows=done.get('rows',[])
    if (done.get('status')!='VERIFIED' or done.get('run_id')!=RUN or done.get('target_access') is not False
        or done.get('public_packets_per_model')!=30 or done.get('paired_cases')!=1440 or len(rows)!=8):
        raise ValueError('Incomplete public matrix or scope')
    seen=set()
    for row in rows:
        key=(row.get('variant'),row.get('model_seed'))
        if key not in expected or key in seen or row.get('row_id')!=expected[key]:raise ValueError('Duplicate or unexpected frozen model')
        if (row.get('status')!='VERIFIED' or row.get('public_packets')!=30 or row.get('cases')!=180
            or row.get('optimizer_steps')!=0 or row.get('source_IQ_access') is not False
            or row.get('target_access') is not False or row.get('model_state_unchanged') is not True):
            raise ValueError('Frozen public row scope differs')
        seen.add(key)
    return rows

def protect_outputs(evidence):
    for name in OUTPUTS:
        if (evidence/name).exists():raise FileExistsError('Preserve existing independent analysis: '+name)

def fft(x,window):
    frames=np.lib.stride_tricks.sliding_window_view(x,64,axis=-1)[...,::32,:]*window
    z=frames[:,0]+1j*frames[:,1]
    s=np.fft.fftshift(np.fft.fft(z,axis=-1),axes=-1)/np.linalg.norm(window)
    return np.stack((s.real,s.imag),1).transpose(0,1,3,2)

def relation(spectra,ar,ai,subspace):
    z=spectra[:,0]+1j*spectra[:,1]
    pair=np.stack((z[:,33:64],z[:,1:32][:,::-1].conj()),-2)
    v=pair@(ar+1j*ai).T;energy=(abs(v)**2).sum((-2,-1))
    floor=np.maximum(energy.mean(-1,keepdims=True)/64,1e-6)
    denom=np.maximum(energy,floor);n=v/np.sqrt(denom)[...,None,None]
    w=n[...,0,:,None]*n[...,1,None,:]-n[...,1,:,None]*n[...,0,None,:]
    det=.5*(abs(w)**2).sum((-2,-1))
    if subspace:q=w.conj()@w.swapaxes(-1,-2)*(energy/denom/(2*np.maximum(det,1e-3)))[...,None,None]
    else:q=n.conj().swapaxes(-1,-2)@n
    return np.stack((q.real,q.imag),1),energy<floor,det<1e-3

def validate_arrays(a,subspace):
    h=a['history_iq'].astype(float)
    if h.shape!=(30,2,272) or any(not np.isfinite(v).all() for v in a.values()):raise ValueError('Public tensor schema/finite check failed')
    if not np.allclose(a['window'],np.hanning(65)[:-1],atol=3e-7,rtol=0):raise ValueError('Periodic window differs')
    checks=[]
    for precision in ('fp32','fp64'):
        tol=5e-5 if precision=='fp32' else 5e-9
        base=a[precision+'_spectra_base']
        if np.max(abs(fft(h[...,16:],a['window'].astype(float))-base))>tol:raise ValueError('Independent base STFT differs')
        for delay in DELAYS:
            for mode in ('zero','continuous'):
                key=f'{precision}_d{delay}_{mode}_'
                past=h.copy()
                if mode=='zero':past[...,:16]=0
                delayed=past[...,16-delay:272-delay]
                raw=h[...,16:]+np.stack((.23*delayed[:,0]-.09*delayed[:,1],.09*delayed[:,0]+.23*delayed[:,1]),1)
                raw_error=float(np.max(abs(raw-a[key+'raw_iq'])))
                normalized=raw/np.sqrt(np.maximum((raw**2).sum(1).mean(-1),1e-12))[:,None,None]
                if raw_error>tol or np.max(abs(normalized-a[key+'iq']))>tol:raise ValueError('Independent paired FIR/RMS differs')
                if np.max(abs(fft(a[key+'raw_iq'],a['window'].astype(float))-a[key+'raw_spectra']))>tol:raise ValueError('Raw FFT differs')
                if np.max(abs(fft(a[key+'iq'],a['window'].astype(float))-a[key+'spectra']))>tol:raise ValueError('Normalized FFT differs')
                q,ef,df=relation(a[key+'spectra'].astype(float),a['mix_real'].astype(float),a['mix_imag'].astype(float),subspace)
                err=float(np.max(abs(q-a[key+'q'])))
                if err>tol:raise ValueError('Independent relation formula differs:'+str(err))
                checks.append(dict(precision=precision,delay=delay,mode=mode,FIR_max_error=raw_error,Q_max_error=err,
                    energy_floor_disagreements=int((ef!=a[key+'energy_floor']).sum()),det_floor_disagreements=int((df!=a[key+'determinant_floor']).sum())))
            zero=a[f'{precision}_d{delay}_zero_raw_iq'];continuous=a[f'{precision}_d{delay}_continuous_raw_iq']
            if not np.array_equal(zero[...,delay:],continuous[...,delay:]):raise ValueError('History changed samples after FIR support')
            z=a[f'{precision}_d{delay}_zero_raw_spectra'];c=a[f'{precision}_d{delay}_continuous_raw_spectra']
            if not np.array_equal(z[...,1:],c[...,1:]):raise ValueError('History affected nonboundary raw STFT frames')
        q,_,_=relation(base.astype(float),a['mix_real'].astype(float),a['mix_imag'].astype(float),subspace)
        if np.max(abs(q-a[precision+'_q_base']))>tol:raise ValueError('Independent base relation differs')
    return checks

def records(a,variant,seed):
    packets=[];frequencies=[];frames=[];embeddings=[]
    norm=lambda x:np.linalg.norm(x.reshape(len(x),-1),axis=1)
    for precision in ('fp32','fp64'):
        base=a[precision+'_q_base'].astype(float)
        for delay in DELAYS:
            zero=a[f'{precision}_d{delay}_zero_q'].astype(float);hist=a[f'{precision}_d{delay}_continuous_q'].astype(float)
            dz=zero-base;dh=hist-base;boundary=zero-hist
            square=norm(dz)**2;terms=norm(dh)**2+norm(boundary)**2+2*(dh*boundary).sum((1,2,3,4))
            if not np.allclose(square,terms,atol=1e-11,rtol=1e-11):raise ValueError('Vector decomposition differs')
            for i in range(30):
                packets.append(dict(variant=variant,seed=seed,precision=precision,delay=delay,packet=i,group=GROUPS[i//6],
                    base_Q_norm=norm(base)[i],zero_Q_absolute=norm(dz)[i],continuous_Q_absolute=norm(dh)[i],boundary_Q_absolute=norm(boundary)[i],
                    squared_cross_term=2*float((dh[i]*boundary[i]).sum()),decomposition_error=abs(square[i]-terms[i])))
                for f in range(31):
                    frequencies.append(dict(variant=variant,seed=seed,precision=precision,delay=delay,packet=i,group=GROUPS[i//6],frequency_pair=f+1,
                        zero_Q_absolute=float(np.linalg.norm(dz[i,:,f])),continuous_Q_absolute=float(np.linalg.norm(dh[i,:,f])),boundary_Q_absolute=float(np.linalg.norm(boundary[i,:,f]))))
                s=a[f'{precision}_d{delay}_zero_raw_spectra'][i]-a[f'{precision}_d{delay}_continuous_raw_spectra'][i]
                for frame in range(7):frames.append(dict(variant=variant,seed=seed,precision=precision,delay=delay,packet=i,group=GROUPS[i//6],frame=frame,raw_spectral_boundary_norm=float(np.linalg.norm(s[...,frame]))))
    for delay in DELAYS:
        base=a['embedding_base'];zero=a[f'embedding_d{delay}_zero'];hist=a[f'embedding_d{delay}_continuous']
        for i in range(30):embeddings.append(dict(variant=variant,seed=seed,delay=delay,packet=i,group=GROUPS[i//6],precision='fp32',
            zero_embedding_distance=float(np.linalg.norm(zero[i]-base[i])),continuous_embedding_distance=float(np.linalg.norm(hist[i]-base[i])),boundary_embedding_distance=float(np.linalg.norm(zero[i]-hist[i]))))
    return packets,frequencies,frames,embeddings

def csvwrite(p,rows):
    opener=gzip.open if p.suffix=='.gz' else open
    with opener(p,'xt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze(root,output):
    done=json.loads((output/'completion.json').read_text(encoding='utf-8'))
    rows=validate_completion(done)
    evidence=root/'automation_reports/CV-SincNet'/RUN/'evidence'
    protect_outputs(evidence)
    packets=[];frequencies=[];frames=[];embeddings=[];checks=[]
    for row in rows:
        with np.load(output/row['row_id']/'paired_arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        checks.append(dict(row_id=row['row_id'],checks=validate_arrays(a,row['variant']=='mirror_subspace')))
        p,f,t,e=records(a,row['variant'],row['model_seed']);packets+=p;frequencies+=f;frames+=t;embeddings+=e
    if (len(packets),len(frequencies),len(frames),len(embeddings))!=(1440,44640,10080,720):raise ValueError('Incomplete public records')
    summary=[]
    for variant in ('mirror_energy','mirror_subspace'):
        for precision in ('fp32','fp64'):
            for delay in DELAYS:
                for group in GROUPS:
                    p=[r for r in packets if (r['variant'],r['precision'],r['delay'],r['group'])==(variant,precision,delay,group)]
                    e=[r for r in embeddings if (r['variant'],r['delay'],r['group'])==(variant,delay,group)]
                    q=dict(variant=variant,precision=precision,delay=delay,group=group,count=len(p))
                    for k in ('base_Q_norm','zero_Q_absolute','continuous_Q_absolute','boundary_Q_absolute','squared_cross_term'):q[k]=statistics.mean(r[k] for r in p)
                    for k in ('zero_embedding_distance','continuous_embedding_distance','boundary_embedding_distance'):q['FP32_'+k]=statistics.mean(r[k] for r in e)
                    summary.append(q)
    evidence.mkdir(exist_ok=True)
    protect_outputs(evidence)
    for name,rows in [('packets.csv',packets),('frequency_pairs.csv.gz',frequencies),('frames.csv',frames),('embeddings.csv',embeddings),('summary.csv',summary)]:csvwrite(evidence/name,rows)
    for name,d in [('summary.json',summary),('independent_recount.json',dict(status='VERIFIED',packets=1440,frequency_pairs=44640,frames=10080,embeddings=720,checks=checks,target_access=False))]:
        with (evidence/name).open('x',encoding='utf-8') as handle:
            handle.write(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='VERIFIED',packets=1440,frequency_pairs=44640,frames=10080,embeddings=720)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.root,a.output)
