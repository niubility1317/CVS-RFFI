"""Small frozen source-RX response correction; not a pure-hardware estimator."""
from dataclasses import dataclass
import numpy as np
from .receiver_response import Response,RXS
from experiments.cvs_rff_physics.known_excitation import TONES

RECIPE=dict(strength=.1,max_gain_db=.5,max_phase_rad=.025,RX_rank=2,
    fit_role='U_s',input_shape=[2,256],single_packet_inference=True,
    model_parameters=0,stored_source_statistics=12)


@dataclass(frozen=True)
class Calibration:
    center:np.ndarray
    basis:np.ndarray

    def __post_init__(self):
        center=np.array(self.center,dtype=np.float64,copy=True);basis=np.array(self.basis,dtype=np.float64,copy=True)
        if center.shape!=(4,) or basis.shape!=(2,4) or not np.isfinite(center).all() or not np.isfinite(basis).all() or not np.allclose(basis@basis.T,np.eye(2),atol=1e-8):raise ValueError('Finite frozen4D center and orthonormal2D basis required')
        center.setflags(write=False);basis.setflags(write=False)
        object.__setattr__(self,'center',center);object.__setattr__(self,'basis',basis)


def fit(coefficients,receivers,*,role):
    if role!='U_s':raise ValueError('Only TX-hidden trainingU may fit this calibration')
    x=np.asarray(coefficients,dtype=np.float64);rx=np.asarray(receivers)
    if x.shape!=(len(rx),4) or not np.isfinite(x).all() or set(rx)!=set(RXS):raise ValueError('Finite full sourceRX coefficient population required')
    means=[];counts={}
    for r in RXS:
        part=x[rx==r]
        if len(part)<2:raise ValueError('Every sourceRX must be represented')
        means.append(part.mean(0));counts[str(r)]=len(part)
    means=np.stack(means);center=means.mean(0);effect=means-center
    _,singular,basis=np.linalg.svd(effect,full_matrices=False)
    if singular[1]<=1e-10:raise ValueError('Two supported response directions required')
    basis=basis[:2]
    for row in basis:
        if row[np.abs(row).argmax()]<0:row*=-1
    calibration=Calibration(center,basis)
    return calibration,dict(fit_role=role,records=len(x),counts_by_RX=counts,
        RX_energy_in_top2=float(np.square(singular[:2]).sum()/np.square(singular).sum()),
        neural_parameters=0,stored_statistics=12,TX_labels_consumed=0)


class Correction:
    def __init__(self,calibration):
        if not isinstance(calibration,Calibration):raise ValueError('Frozen source calibration required')
        self.calibration=calibration;self.response=Response()
        u=TONES/24.;design=np.stack((u,u*u),1);self.offset=design.mean(0);design-=self.offset
        _,r=np.linalg.qr(design)
        frequency=np.fft.fftfreq(256);active=max(abs(TONES//4))/20
        coordinate=np.clip(frequency/active,-1,1)
        basis=(np.stack((coordinate,coordinate*coordinate),1)-self.offset)@np.linalg.inv(r)
        taper=np.clip((.5-np.abs(frequency))/(.5-active),0,1)
        self.frequency_basis=basis*taper[:,None]

    def __call__(self,iq,*,strength=RECIPE['strength']):
        a=np.asarray(iq)
        if a.ndim!=3 or a.shape[1:]!=(2,256) or a.dtype not in (np.float32,np.float64) or not np.isfinite(a).all():raise ValueError('Finite FP32/64 packet IQ required')
        if type(strength) not in (float,int) or not np.isfinite(strength) or not 0<=strength<=.1:raise ValueError('Fixedboundedstrength0to.1 only')
        if strength==0:return a.copy(),dict(max_abs_gain_db=0.,max_abs_phase_rad=0.,mean_quality_gate=0.,identity_control=True)
        coeff,quality=self.response(a)
        centered=coeff-self.calibration.center;projected=(centered@self.calibration.basis.T)@self.calibration.basis
        complex_coeff=projected[:,:2]+1j*projected[:,2:]
        delta=complex_coeff@self.frequency_basis.T
        gate=np.clip(quality[:,0],0,1)*np.clip(quality[:,1],0,1)*np.clip(quality[:,2],0,1)
        denominator=1+strength*gate[:,None]*delta
        # Construct a bounded polar inverse;never divide by a deep response null.
        log_bound=np.log(10)*RECIPE['max_gain_db']/20
        raw_gain=-np.log(np.maximum(np.abs(denominator),1e-12));raw_phase=-np.angle(denominator)
        gain=np.clip(raw_gain,-log_bound,log_bound)
        phase=np.clip(raw_phase,-RECIPE['max_phase_rad'],RECIPE['max_phase_rad'])
        transfer=np.exp(gain+1j*phase)
        z=a[:,0].astype(np.float64)+1j*a[:,1].astype(np.float64)
        corrected=np.fft.ifft(np.fft.fft(z,axis=1)*transfer,axis=1)
        result=np.stack((corrected.real,corrected.imag),1).astype(a.dtype)
        if not np.isfinite(result).all():raise ValueError('Finite bounded corrected IQ required')
        return result,dict(max_abs_gain_db=float(np.max(np.abs(gain))*20/np.log(10)),max_abs_phase_rad=float(np.max(np.abs(phase))),
            mean_quality_gate=float(gate.mean()),gain_bound_fraction=float((np.abs(raw_gain)>log_bound).mean()),
            phase_bound_fraction=float((np.abs(raw_phase)>RECIPE['max_phase_rad']).mean()),identity_control=False,neural_parameters=0,stored_source_statistics=12)
