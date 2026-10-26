"""Fixed L-STF mixed-response statistics; no pure-hardware or identity claim."""
import numpy as np
from experiments.cvs_rff_physics.known_excitation import TONES,template

RXS=(1,3,4,6,8)
DAYS=(1,2,3)


class Response:
    def __init__(self):
        self.bank=np.stack([template(np.arange(20),d/16) for d in range(320)])
        self.bins=TONES//4%20;f=TONES/24.
        b=np.stack((f,f*f),1);b-=b.mean(0);self.basis=np.linalg.qr(b)[0]
        self.ideal=np.fft.fft(self.bank,axis=1)[:,self.bins]

    def __call__(self,iq):
        a=np.asarray(iq,dtype=np.float64)
        if a.ndim!=3 or a.shape[1:]!=(2,256) or not np.isfinite(a).all():raise ValueError('Finite source IQ[B,2,256] required')
        z=a[:,0]+1j*a[:,1];w=z[:,80:160];eps=1e-6
        cross=(w[:,20:]*w[:,:-20].conj()).sum(1);omega=np.arctan2(cross.imag,cross.real+eps**2)/20
        corrected=w*np.exp(-1j*omega[:,None]*np.arange(80));mean=corrected.reshape(-1,4,20).mean(1)
        energy=(abs(mean)**2).mean(1);unit=mean/np.sqrt(energy+eps**2)[:,None]
        corr=unit@self.bank.conj().T/20;chosen=(abs(corr)**2).argmax(1);gain=corr[np.arange(len(a)),chosen]
        observed=np.fft.fft(unit,axis=1)[:,self.bins];ideal=self.ideal[chosen]
        transfer=observed*ideal.conj()/abs(ideal)**2
        relative=transfer*gain.conj()[:,None]/(abs(gain)**2+eps)[:,None]
        residual=relative-1;bounded=residual/np.sqrt(1+abs(residual)**2/16)
        low=bounded@self.basis;coeff=np.concatenate((low.real,low.imag),1)
        power=(abs(w)**2).mean(1)
        norm=np.sqrt((abs(w[:,20:])**2).sum(1)*(abs(w[:,:-20])**2).sum(1)+eps**4)
        quality=np.stack((abs(gain)**2,abs(cross)/norm,energy/(power+eps**2),omega*20/np.pi),1)
        if not np.isfinite(coeff).all() or not np.isfinite(quality).all():raise ValueError('Nonfinite mixed response')
        return coeff,quality


def agreement(fit,held):
    fit=np.asarray(fit);held=np.asarray(held);norm=float(np.square(held).sum());other=float(np.square(fit).sum())
    return dict(r2=1-float(np.square(held-fit).sum())/norm if norm>1e-20 else None,
        cosine=float((fit*held).sum())/np.sqrt(norm*other) if min(norm,other)>1e-20 else None)


def decompose(coeff,tx,rx,day):
    if coeff.shape!=(len(tx),4) or any(len(v)!=len(tx) for v in (rx,day)) or set(tx)!={0,1,2,3,4,5} or set(rx)!=set(RXS) or set(day)!=set(DAYS):
        raise ValueError('Complete true-L sixTX/fiveRX/threeDay response required')
    means=np.empty((6,5,3,4));counts=np.empty((6,5,3),np.int64);within=0.
    for t in range(6):
        for ri,r in enumerate(RXS):
            for di,d in enumerate(DAYS):
                v=coeff[(tx==t)&(rx==r)&(day==d)];counts[t,ri,di]=len(v)
                if len(v)<2:raise ValueError('Each labeled source cell must be observed')
                means[t,ri,di]=v.mean(0);within+=float(np.square(v-means[t,ri,di]).sum())
    if counts.min()!=counts.max():raise ValueError('Balanced labeled cells required for orthogonal variance decomposition')
    grand=means.mean((0,1,2));a=means.mean((1,2))-grand;b=means.mean((0,2))-grand;c=means.mean((0,1))-grand
    interaction=means-grand-a[:,None,None,:]-b[None,:,None,:]-c[None,None,:,:]
    total=float(np.square(coeff-grand).sum()/len(coeff));parts=dict(TX=float(np.square(a).sum()/6),RX=float(np.square(b).sum()/5),
        day=float(np.square(c).sum()/3),interaction=float(np.square(interaction).sum()/90),within_cell_packet=within/len(coeff))
    if abs(sum(parts.values())-total)>1e-9*max(1,total):raise ValueError('Balanced variance identity failed')
    folds=[]
    for t in range(6):
        for di,d in enumerate(DAYS):
            train=means[np.arange(6)!=t][:,:,np.arange(3)!=di]
            fit=train.mean((0,2));fit-=fit.mean(0)
            held=means[t,:,di];held=held-held.mean(0)
            folds.append(dict(held_TX=t,held_day=d,train_TX=5,train_days=2,train_packets=int(counts[0,0,0])*50,
                held_packets=int(counts[0,0,0])*5,**agreement(fit,held)))
    rx_basis=np.linalg.svd(b,full_matrices=False)[2][:2];tx_basis=np.linalg.svd(a,full_matrices=False)[2][:2]
    strength=parts['RX']/total if total>1e-20 else 0.
    r2=[r['r2'] for r in folds if r['r2'] is not None]
    necessary=dict(all_folds_defined=len(r2)==18,median_crossed_r2_atleast_quarter=len(r2)==18 and float(np.median(r2))>=.25,
        positive_crossed_folds_atleast15=sum(v>0 for v in r2)>=15,receiver_fraction_atleast_one_percent=strength>=.01)
    result=dict(cells=90,cell_min=int(counts.min()),cell_max=int(counts.max()),features=4,variance=parts,total_variance=total,
        RX_fraction=strength,crossed_folds=folds,necessary_conditions=necessary,common_component_supported=all(necessary.values()),
        TX_energy_in_RX_top2=float(np.square(a@rx_basis.T).sum()/np.square(a).sum()) if np.square(a).sum()>1e-20 else None,
        RX_energy_in_top2=float(np.square(b@rx_basis.T).sum()/np.square(b).sum()) if np.square(b).sum()>1e-20 else None,
        interpretation='mixed received response;stable sourceRX effects do not identify pure hardware or an unseenRX estimator;TXmean preservation under zero-mean subtraction is algebraic,not identity protection')
    return result,dict(cell_means=means,cell_counts=counts,RX_effect=b,TX_effect=a,RX_top2=rx_basis,TX_top2=tx_basis)


class PairMoments:
    def __init__(self,rx_basis,tx_basis):self.rx_basis=rx_basis;self.tx_basis=tx_basis;self.groups={}
    def update(self,raw,perturbed,rx,scene):
        if raw.shape!=perturbed.shape or raw.shape!=(len(rx),4) or set(rx)-set(RXS):raise ValueError('Aligned hidden-U same-packet source RX pairs required')
        delta=perturbed-raw
        for r in np.unique(rx):
            v=delta[rx==r];g=self.groups.setdefault((scene,int(r)),dict(count=0,energy=0.,RX_energy=0.,TX_energy=0.))
            g['count']+=len(v);g['energy']+=float(np.square(v).sum());g['RX_energy']+=float(np.square(v@self.rx_basis.T).sum());g['TX_energy']+=float(np.square(v@self.tx_basis.T).sum())
    def result(self):
        return [dict(scene=s,RX=r,count=g['count'],mean_squared_change=g['energy']/g['count'],
            fraction_in_RX_top2=g['RX_energy']/g['energy'] if g['energy']>1e-20 else None,
            fraction_in_TX_top2=g['TX_energy']/g['energy'] if g['energy']>1e-20 else None) for (s,r),g in sorted(self.groups.items())]
