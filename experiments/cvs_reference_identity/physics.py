"""Fixed TX -> channel -> RX -> bandlimit/ADC mechanism diagnostic, not augmentation.

The steady L-STF is evaluated at100Msps to avoid aliasing cubic TX products
before the declared in-band projection/25Msps sampling. This is a simplified
periodic receiver, not a reconstruction of the WiSig MATLAB equalizer.
"""
import numpy as np
import torch
from torch.nn import functional as F
from experiments.cvs_rff_physics.known_excitation import TONES,SIGNS,REFERENCE

TX_ROWS=(
    dict(name='ideal',cubic=0j,image=0j,memory_cubic=0j),
    dict(name='tx_am_am',cubic=-.03+0j,image=0j,memory_cubic=0j),
    dict(name='tx_first_order_am_pm',cubic=.03j,image=0j,memory_cubic=0j),
    dict(name='tx_iq_image',cubic=0j,image=.02+.02j,memory_cubic=0j),
    dict(name='tx_cubic_memory',cubic=0j,image=0j,memory_cubic=.02j),
)
RX_ROWS=(
    dict(name='identity',gain=1+0j,tap=0j,delay=1,image=0j,cubic=0j,cfo_hz=0.),
    dict(name='flat_phase_gain',gain=1.3*np.exp(.73j),tap=0j,delay=1,image=0j,cubic=0j,cfo_hz=0.),
    dict(name='relative_cfo',gain=1+0j,tap=0j,delay=1,image=0j,cubic=0j,cfo_hz=80_000.),
    dict(name='lti_channel',gain=1+0j,tap=.15+.08j,delay=1,image=0j,cubic=0j,cfo_hz=0.),
    dict(name='rx_iq_image',gain=1+0j,tap=0j,delay=1,image=.02+.02j,cubic=0j,cfo_hz=0.),
    dict(name='rx_cubic',gain=1+0j,tap=0j,delay=1,image=0j,cubic=-.03+0j,cfo_hz=0.),
)


def highrate_reference():
    n=np.arange(80,dtype=np.float64)
    return np.sum(SIGNS*(1+1j)*np.exp(2j*np.pi*n[:,None]*TONES/320),axis=1)/np.sqrt(24)


def received(tx,rx):
    s=highrate_reference();cubic=s*abs(s)**2
    # Native25Msps delay2 equals8 samples at100Msps. Steady periodic history.
    transmitter=s+tx['cubic']*cubic+tx['image']*s.conjugate()+tx['memory_cubic']*np.roll(cubic,8)
    propagated=transmitter+rx['tap']*np.roll(transmitter,4*rx['delay'])
    receiver=rx['gain']*propagated+rx['image']*propagated.conjugate()+rx['cubic']*propagated*abs(propagated)**2
    spectrum=np.fft.fft(receiver)
    keep=np.zeros(80,dtype=bool);keep[TONES//4%80]=True
    inband=np.fft.ifft(np.where(keep,spectrum,0))[::4]
    z=np.tile(inband,13)[:256]
    z*=np.exp(2j*np.pi*rx['cfo_hz']*np.arange(256)/25_000_000)
    z/=np.sqrt(np.mean(abs(z)**2))
    return np.stack([z.real,z.imag])


def controlled_cascade_diagnostics(model):
    model.eval();parameter=next(model.parameters())
    arrays=[received(t,r) for t in TX_ROWS for r in RX_ROWS]
    x=torch.tensor(np.stack(arrays).tolist(),dtype=parameter.dtype,device=parameter.device)
    features=F.normalize(model.features(x),dim=1,eps=1e-4)
    logits=model.classify_features(model.features(x))
    response=model.response.components(x)
    transfer=response['relative_transfer']
    records=[]
    for ti,tx in enumerate(TX_ROWS):
        for ri,rx in enumerate(RX_ROWS):
            index=ti*len(RX_ROWS)+ri;base=ti*len(RX_ROWS)
            records.append(dict(tx=tx['name'],rx=rx['name'],
                unit_embedding_distance_same_tx_identity_rx=float((features[index]-features[base]).norm()),
                relative_transfer_distance_same_tx_identity_rx=float((transfer[index]-transfer[base]).abs().square().mean().sqrt()),
                response_cfo_hz=float(response['cfo'][index]*625_000),
                source_classifier_logits=logits[index].tolist()))
    pairs=[]
    for ti in range(1,len(TX_ROWS)):
        base=ti*len(RX_ROWS)
        pairs.append(dict(tx=TX_ROWS[ti]['name'],rx='identity',
            unit_embedding_distance_to_ideal_tx=float((features[base]-features[0]).norm()),
            relative_transfer_distance_to_ideal_tx=float((transfer[base]-transfer[0]).abs().square().mean().sqrt())))
    # At unity channel TX image vs RX image and TX cubic vs RX cubic coincide.
    iq_error=float(np.max(abs(received(TX_ROWS[3],RX_ROWS[0])-received(TX_ROWS[0],RX_ROWS[4]))))
    cubic_error=float(np.max(abs(received(TX_ROWS[1],RX_ROWS[0])-received(TX_ROWS[0],RX_ROWS[5]))))
    if max(iq_error,cubic_error)>1e-12:raise AssertionError('Declared TX/RX confounds must coincide')
    return dict(status='FROZEN_SYNTHETIC_DIAGNOSTIC_COMPLETE',reference=REFERENCE,
        synthetic_only=True,training_augmentation=False,formal_data_access=False,target_access=False,
        model_updated=False,hardware_parameter_recovery=False,identity_accuracy='N/A: synthetic settings are not source TX classes',
        simulated_chain='100Msps steady L-STF -> TX cubic/image/memory -> causal periodic FIR channel -> RX gain/image/cubic -> retain12occupiedtones -> decimate25Msps -> relative CFO -> packetRMS',
        exact_wisig_equalizer=False,onset_transient_simulated=False,noise_simulated=False,
        physical_parameters=dict(tx=[{k:([v.real,v.imag] if isinstance(v,complex) else v) for k,v in t.items()} for t in TX_ROWS],
            rx=[{k:([v.real,v.imag] if isinstance(v,complex) else v) for k,v in r.items()} for r in RX_ROWS],tx_memory_delay_native_samples=2),
        records=records,isolated_tx_changes=pairs,
        confounds=dict(iq_tx_rx_waveform_max_error=iq_error,cubic_tx_rx_waveform_max_error=cubic_error),
        learned_response_gain_abs_mean=float(model.response_gain.detach().tanh().abs().mean()),
        claim='Whole embedding sensitivity under fixed TX/RX cascade, with identical-observation counterexamples; no calibrated TX or universal RX invariance claim')
