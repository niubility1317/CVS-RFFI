"""Training-only, group-constant mixed-response shifts; no hardware separation."""
from dataclasses import dataclass
import numpy as np
from .receiver_response import RXS
from experiments.cvs_rff_physics.known_excitation import TONES

RECIPE=dict(strength=.1,max_gain_db=.5,max_phase_rad=.025,fit_role='U_s',
    stored_training_statistics=20,new_inference_parameters=0,inference='unchanged_raw_F',
    per_packet_response_subtraction=False,quality_gate=False)


@dataclass(frozen=True)
class MeanBank:
    means:np.ndarray

    def __post_init__(self):
        values=np.array(self.means,dtype=np.float64,copy=True)
        if values.shape!=(5,4) or not np.isfinite(values).all():raise ValueError('Five frozen sourceRX means of four coefficients required')
        values.setflags(write=False);object.__setattr__(self,'means',values)


def fit(coefficients,receivers,*,role):
    x=np.asarray(coefficients,dtype=np.float64);rx=np.asarray(receivers)
    if role!='U_s' or x.shape!=(len(rx),4) or not np.isfinite(x).all() or set(rx)!=set(RXS):raise ValueError('Only complete TX-hidden sourceU/RX coefficients may fit mean bank')
    counts={str(r):int((rx==r).sum()) for r in RXS}
    if min(counts.values())<2:raise ValueError('Every source receiver needs observations')
    return MeanBank(np.stack([x[rx==r].mean(0) for r in RXS])),counts


def transfers(bank):
    if not isinstance(bank,MeanBank):raise ValueError('Frozen sourceU means required')
    u=TONES/24.;design=np.stack((u,u*u),1);offset=design.mean(0);design-=offset
    _,r=np.linalg.qr(design);frequency=np.fft.fftfreq(256);active=max(abs(TONES//4))/20
    coordinate=np.clip(frequency/active,-1,1)
    basis=(np.stack((coordinate,coordinate*coordinate),1)-offset)@np.linalg.inv(r)
    basis*=np.clip((.5-np.abs(frequency))/(.5-active),0,1)[:,None]
    # [original RX, style RX, coefficient]. Shift the group mean; do not remove
    # the current packet's measured response or assign it a new physical RX.
    difference=bank.means[None,:,:]-bank.means[:,None,:]
    delta=(difference[...,:2]+1j*difference[...,2:])@basis.T
    action=1+RECIPE['strength']*delta
    bound=np.log(10)*RECIPE['max_gain_db']/20
    gain=np.clip(np.log(np.maximum(abs(action),1e-12)),-bound,bound)
    phase=np.clip(np.angle(action),-RECIPE['max_phase_rad'],RECIPE['max_phase_rad'])
    values=np.exp(gain+1j*phase).astype(np.complex64)
    return values,dict(max_abs_gain_db=float(abs(gain).max()*20/np.log(10)),max_abs_phase_rad=float(abs(phase).max()),
        gain_bound_fraction=float((abs(np.log(np.maximum(abs(action),1e-12)))>bound).mean()),
        phase_bound_fraction=float((abs(np.angle(action))>RECIPE['max_phase_rad']).mean()))


class Shift:
    def __init__(self,bank,device):
        import torch
        values,self.diagnostic=transfers(bank)
        # Keep the bank outside nn.Module/state_dict: deployment uses original F.
        self.transfer=torch.frombuffer(bytearray(values.tobytes()),dtype=torch.complex64).reshape(5,5,256).clone().to(device)
        self.rxs=torch.tensor(RXS,device=device)

    def __call__(self,iq,receivers,physical_indices,call):
        import torch
        if iq.ndim!=3 or tuple(iq.shape[1:])!=(2,256) or iq.dtype!=torch.float32 or not torch.isfinite(iq).all():raise ValueError('Finite FP32 source packet IQ required')
        rx=receivers.to(iq.device);index=physical_indices.to(iq.device)
        matches=rx[:,None]==self.rxs[None,:]
        if len(rx)!=len(iq) or len(index)!=len(iq) or not bool(matches.sum(1).eq(1).all()) or type(call) is not int or call<1:raise ValueError('Original sourceRX, physical indices and ordered training call required')
        original=matches.long().argmax(1);style=(original+1+(index+call-1).remainder(4)).remainder(5)
        z=torch.complex(iq[:,0],iq[:,1]);out=torch.fft.ifft(torch.fft.fft(z,dim=1)*self.transfer[original,style],dim=1)
        return torch.stack((out.real,out.imag),1),self.rxs[style]
