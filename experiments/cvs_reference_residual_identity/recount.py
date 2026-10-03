"""Independent NumPy solve and complete public-cascade evidence verification."""
import itertools,json
from pathlib import Path
import numpy as np
from experiments.cvs_rff_physics.known_excitation import template
from .physics import OUT,TX,CHANNELS,RX,write
from .model import VARIANTS

def numpy_components(iq,variant,ridge):
    x=iq[:,0].astype(np.float64)+1j*iq[:,1].astype(np.float64);w=x[:,80:160]
    cross=(w[:,20:]*w[:,:-20].conj()).sum(1);omega=np.arctan2(cross.imag,cross.real+1e-12)/20
    w=w*np.exp(-1j*omega[:,None]*np.arange(80));w=w/np.sqrt(np.mean(abs(w)**2,axis=1)+1e-12)[:,None]
    cycles=w.reshape(-1,4,20);bank=np.stack([template(np.arange(20),d/16) for d in range(320)])
    correlations=cycles.mean(1)@bank.conj().T/20;chosen=(abs(correlations)**2).argmax(1)
    y=np.fft.fft(cycles,axis=-1)/20;s=np.fft.fft(bank[chosen],axis=-1)/20;m=y.mean(1)
    d=np.exp(-2j*np.pi*np.arange(20)[:,None]*np.arange(-4,5)[None]/20)
    a=s[:,:,None]*d[None];gram=np.einsum('bfi,bfj->bij',a.conj(),a)
    rhs=np.einsum('bfi,bf->bi',a.conj(),m)
    h=np.linalg.solve(gram+np.diag(ridge)[None],rhs[...,None]).squeeze(-1);response=np.einsum('fl,bl->bf',d,h)
    residual=y-(s*response)[:,None];power=(abs(response)**2).mean(1,keepdims=True)
    gain=(s.conj()*m).sum(1)/((abs(s)**2).sum(1)+1e-12)
    denom=response if variant==VARIANTS[1] else np.broadcast_to(gain[:,None],response.shape)
    inverse=denom.conj()/(abs(denom)**2+.01*power+1e-8);canonical=residual*inverse[:,None]
    v=np.fft.ifft(canonical*20,axis=-1).reshape(-1,80);v=v/np.sqrt(1+abs(v)**2/4)
    return dict(features=np.stack((v.real,v.imag),1),cycle_spectra=y,reference_spectrum=s,response=response,coefficients=h,
        residual_spectra=residual,canonical_spectra=canonical,selected_delay=chosen/16,
        fit_error=np.mean(abs(residual)**2,axis=(1,2)),reference_quality=np.max(abs(correlations)**2,axis=1),
        inverse_abs_max=np.max(abs(inverse),axis=1),relative_cfo=omega*20/np.pi)

def independent_inputs(records):
    n=np.arange(256)
    def tx_at(n,name):
        s=template(n);old=template(n-1)
        return dict(ideal=s,linear_d1=s+(.08+.03j)*old,iq_mirror=s+(.06+.02j)*s.conj(),
            pa3=s+(.04+.01j)*s*abs(s)**2,pa5=s+(.007+.003j)*s*abs(s)**4,memory3=s+(.04+.01j)*old*abs(old)**2)[name]
    values=[]
    for r in records:
        v=tx_at(n,r['tx'])
        if r['channel']!='identity':v=v+(.23+.09j)*tx_at(n-int(r['channel'][5:]),r['tx'])
        if r['rx']=='iq_mirror':v=v+(.04-.01j)*v.conj()
        if r['rx']=='pa3':v=v+(.015+.01j)*v*abs(v)**2
        v=v/np.sqrt(np.mean(abs(v)**2));values.append([v.real,v.imag])
    return np.asarray(values,dtype=np.float32)

def recount():
    read=lambda name:json.loads((OUT/name).read_text(encoding='utf-8'))
    if (OUT/'independent_recount.json').exists():raise FileExistsError('Preserve existing verification')
    done=read('complete.json');expected=dict(status='ARTIFACTS_COMPLETE',variants=2,public_inputs_per_variant=72,TX_signature_pairs=120,target_access=False,training=False)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Incomplete cascade')
    records=read('records.json');expect=[dict(tx=t,channel=h,rx=r) for t,h,r in itertools.product(TX,CHANNELS,RX)]
    if records!=expect:raise ValueError('Fixed cascade record matrix differs')
    iq=np.load(OUT/'inputs.npz')['iq']
    if not np.array_equal(iq,independent_inputs(records)):raise ValueError('Public transmitted/channel/receiver cascade differs')
    rows=read('metrics.json')['rows'];keys={(r['variant'],r['tx'],r['channel'],r['rx']) for r in rows}
    if keys!=set(itertools.product(VARIANTS,TX[1:],CHANNELS,RX)) or len(rows)!=120:raise ValueError('Incomplete/duplicate signature matrix')
    lookup={(r['tx'],r['channel'],r['rx']):i for i,r in enumerate(records)};errors={};metric_errors=[]
    for variant in VARIANTS:
        arrays=np.load(OUT/(variant+'.npz'));ridge=arrays['ridge']
        if ridge.shape!=(9,) or not np.isfinite(ridge).all() or not np.allclose(ridge,.001,rtol=1e-6,atol=1e-10):raise ValueError('Not fixed untrained ridge initialization')
        independent=numpy_components(iq,variant,ridge)
        for key,value in independent.items():
            actual=arrays[key]
            if actual.shape!=value.shape or not np.isfinite(actual).all() or not np.isfinite(value).all():raise ValueError('Array schema/nonfinite mismatch')
            if key=='selected_delay':
                if not np.array_equal(actual,value):raise ValueError('Delay-bank routing differs')
            elif not np.allclose(actual,value,rtol=1e-6 if key=='features' else 1e-9,atol=1e-7 if key=='features' else 1e-9):raise ValueError('Independent NumPy solver differs: '+key)
            errors[variant+'/'+key]=float(np.max(abs(actual-value)))
        feature=arrays['features'].astype(np.float64)
        for row in [r for r in rows if r['variant']==variant]:
            tx,channel,rx=(row[k] for k in ('tx','channel','rx'));i=lookup[(tx,channel,rx)];b=lookup[('ideal',channel,rx)]
            anchor=lookup[(tx,'identity','identity')];anchorbase=lookup[('ideal','identity','identity')]
            delta=(feature[i]-feature[b]).ravel();ref=(feature[anchor]-feature[anchorbase]).ravel();raw=(iq[i].astype(np.float64)-iq[b]).ravel()
            size=np.linalg.norm(delta)/np.sqrt(160);rawsize=np.linalg.norm(raw)/np.sqrt(512);refsize=np.linalg.norm(ref)/np.sqrt(160)
            values=dict(TX_signature_rms=size,raw_TX_difference_rms=rawsize,TX_retention_ratio=size/rawsize if rawsize>1e-8 else None,
                relative_signature_drift=(np.linalg.norm(delta-ref)/np.sqrt(160))/refsize if refsize>1e-8 else None)
            for key,value in values.items():
                if value is None:
                    if row[key] is not None:raise ValueError('Undefined metric must be null')
                elif not np.isfinite(row[key]) or not np.isclose(value,row[key],rtol=1e-10,atol=1e-12):raise ValueError('Signature metric differs')
                else:metric_errors.append(abs(float(value)-row[key]))
    write(OUT/'independent_recount.json',dict(status='VERIFIED',public_inputs=72,operator_rows=144,signature_pairs=120,
        numpy_array_max_abs_errors=errors,signature_metric_max_error=max(metric_errors),target_access=False,training=False))
    print('VERIFIED NumPy forward and all120 signature comparisons')

if __name__=='__main__':recount()
