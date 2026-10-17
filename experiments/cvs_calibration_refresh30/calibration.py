"""Refresh only existing injected receiver calibration errors; never alter true hardware."""
from dataclasses import replace
import numpy as np

def refresh(cfg,receiver,epoch,seed):
 from leo_practical.channel import stable_seed
 if epoch not in range(21,31) or isinstance(epoch,bool) or int(epoch)!=epoch:raise ValueError('Cumulative epoch21..30 only')
 if not receiver.session_id or cfg.iq_estimation_amp_std_db!=.05 or cfg.iq_estimation_phase_std_deg!=.2 or not cfg.iq_compensation_enabled:raise ValueError('Original calibrated receiver distribution required')
 rng=np.random.default_rng(stable_seed(seed,'RX_calibration',receiver.session_id,int(epoch)))
 return replace(receiver,estimated_amplitude_imbalance_db=receiver.amplitude_imbalance_db+float(rng.normal(0,cfg.iq_estimation_amp_std_db)),estimated_quadrature_error_deg=receiver.quadrature_error_deg+float(rng.normal(0,cfg.iq_estimation_phase_std_deg)))
