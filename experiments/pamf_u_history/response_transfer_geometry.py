"""Receiver-table algebra only; no waveform, TX, network, or identity guarantee."""
import itertools
import numpy as np


def validate(values):
    v=np.asarray(values,dtype=np.complex128)
    if v.ndim!=3 or v.shape[:2]!=(5,5) or not v.shape[2] or not np.isfinite(v).all() or (abs(v)==0).any():raise ValueError('Finite nonzero five-receiver complex transfer table required')
    if abs(np.angle(v)).max()>=np.pi/3:raise ValueError('Small source response actions required for unambiguous three-edge log phases')
    return v


def cycle_statistics(v):
    v=validate(v);n=5
    reverse=np.stack([v[r,s]*v[s,r] for r in range(n) for s in range(n) if r!=s])
    triangles=np.stack([v[r,s]*v[s,t]*v[t,r] for r,s,t in itertools.permutations(range(n),3)])
    def summarize(a):
        gain=abs(20*np.log10(abs(a)));phase=abs(np.angle(a));error=abs(a-1)
        return dict(paths=len(a),frequency_bins=a.shape[1],max_absolute_gain_error_db=float(gain.max()),max_absolute_phase_error_rad=float(phase.max()),
            RMS_complex_error=float(np.sqrt(np.square(error).mean())),max_complex_error=float(error.max()),
            absolute_gain_error_db_p95=float(np.quantile(gain,.95)),absolute_phase_error_rad_p95=float(np.quantile(phase,.95)))
    z=np.log(abs(v))+1j*np.angle(v);potential=(z.sum(0)-z.sum(1))/(2*n);fitted=potential[None,:,:]-potential[:,None,:];residual=z-fitted
    def energy_fraction(a,b):
        denominator=float(np.square(a).sum());return float(np.square(b).sum()/denominator) if denominator>1e-30 else 0.
    return dict(receiver_nodes=5,directed_pairs=20,identity_max_complex_error=float(abs(v[np.arange(5),np.arange(5)]-1).max()),
        per_edge_max_abs_gain_db=float(abs(20*np.log10(abs(v))).max()),per_edge_max_abs_phase_rad=float(abs(np.angle(v)).max()),
        reciprocal=summarize(reverse),triangles=summarize(triangles),
        nonconservative_log_gain_energy_fraction=energy_fraction(z.real,residual.real),nonconservative_phase_energy_fraction=energy_fraction(z.imag,residual.imag),
        uniform_frequency_table_only=True,IQ_or_identity_or_F_performance_claim=False)


def conservative_projection(values,*,max_gain_db=.5,max_phase_rad=.025):
    v=validate(values)
    if not np.isfinite(max_gain_db) or not np.isfinite(max_phase_rad) or max_gain_db<=0 or not 0<max_phase_rad<np.pi/3:raise ValueError('Finite positive source transfer bounds required')
    z=np.log(abs(v))+1j*np.angle(v);potential=(z.sum(0)-z.sum(1))/10
    # A shared potential per receiver makes every relative edge reciprocal and
    # cycle-consistent. Range limits are shared across RX, not pairwise clips.
    def bounded(a,bound):
        width=np.ptp(a,axis=0);scale=np.minimum(1.,bound/np.maximum(width,1e-30))
        centre=(a.max(0)+a.min(0))/2;return (a-centre)*scale,float((scale<1).mean())
    gain,gain_scaled=bounded(potential.real,np.log(10)*max_gain_db/20);phase,phase_scaled=bounded(potential.imag,max_phase_rad)
    potential=gain+1j*phase;result=np.exp(potential[None,:,:]-potential[:,None,:])
    return result,dict(gain_frequency_range_scaled_fraction=gain_scaled,phase_frequency_range_scaled_fraction=phase_scaled,
        derived_only_from_existing_table=True,new_training_statistics=0,new_F_forwards=0,identity_preservation_guarantee=False)
