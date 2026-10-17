"""Fixed synthetic full-context records. No radio recordings or identity labels."""
import numpy as np
from . import design as d
START=128
FULL_LENGTH=512
def records():
    rng=np.random.default_rng(d.SEED);items=[]
    t=np.arange(FULL_LENGTH,dtype=np.float64)
    def noise():return (rng.normal(size=FULL_LENGTH)+1j*rng.normal(size=FULL_LENGTH))/np.sqrt(2)
    for family in d.FAMILIES:
        for j in range(4):
            if family=="white":full=noise()
            elif family=="colored":
                z=noise();full=np.empty_like(z);previous=0j
                for n in range(FULL_LENGTH):previous=.8*previous+.6*z[n];full[n]=previous
            elif family=="qpsk":
                q=np.exp(1j*(np.pi/4+np.pi/2*rng.integers(0,4,FULL_LENGTH)))
                full=np.convolve(q,[.25,.5,.25],mode="same")+.01*noise()
            elif family=="chirp":
                full=np.exp(1j*(.12*(j+1)*t+.001*(j+1)*t*t))+.03*noise()
            elif family=="periodic":
                base=np.exp(1j*(np.pi/4+np.pi/2*rng.integers(0,4,16)))
                full=np.tile(base,FULL_LENGTH//16)*np.exp(2j*np.pi*.0001*t)+.03*noise()
            elif family=="edge_burst":
                center=(START+2,START+6,START+d.LENGTH-3,START+d.LENGTH-7)[j]
                full=np.exp(-.5*((t-center)/3.)**2)*np.exp(1j*(.3+.2*j)*t)
            elif family=="amplitude_step":
                cut=(START,START+1,START+d.LENGTH-1,START+d.LENGTH//2)[j]
                full=np.where(t<cut,.2,1.)*np.exp(1j*.07*(j+1)*t)
            elif family=="edge_impulse":
                full=np.zeros(FULL_LENGTH,np.complex128)
                pos=(START,START+1,START+d.LENGTH-1,START+d.LENGTH-2)[j]
                full[pos]=np.exp(1j*.37*j)
            else:raise ValueError("Unregistered synthetic family")
            raw=full[START:START+d.LENGTH];power=float(np.mean(np.abs(raw)**2))
            if power<=0:raise ValueError("Invalid synthetic record")
            full=full/np.sqrt(power)
            # Quantize the complete known context as the observed IQ would be.
            full=full.real.astype(np.float32).astype(np.float64)+1j*full.imag.astype(np.float32).astype(np.float64)
            raw=full[START:START+d.LENGTH]
            iq=np.stack((raw.real,raw.imag)).astype(np.float32)[None]
            items.append(dict(case_id=family+"_"+str(j),family=family,index=j,full=full,iq=iq))
    if len(items)!=32 or len({r["case_id"] for r in items})!=32:raise ValueError("Fixed 32 cases required")
    return items
def dense_reference(full,a,start=START):
    """Independent FIR impulse-series approximation, not the recurrence."""
    h=np.empty(96,np.float64);h[0]=a
    h[1:]=(1-a*a)*np.power(-a,np.arange(95))
    y=np.convolve(np.asarray(full,np.complex128),h,mode="full")
    z=y[start+1:start+1+d.LENGTH]
    return np.stack((z.real,z.imag)).astype(np.float32)[None]
def reflected_reference(iq,a):
    z=iq[0,0].astype(np.float64)+1j*iq[0,1].astype(np.float64)
    z=np.pad(z,(d.CONTEXT,d.CONTEXT),mode="reflect")
    return dense_reference(z,a,start=d.CONTEXT)
