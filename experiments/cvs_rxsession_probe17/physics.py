"""Change only injected receiver session;underlying physical/noise draw retained."""
from dataclasses import asdict
import math
import numpy as np
from . import design as d

def session(rx,day,eq,bank):
 if type(rx) is not int or rx not in d.FIT or type(day) is not int or day not in d.DAYS or type(eq) is not int or eq!=1 or bank not in d.BANKS:raise ValueError('Only fixed fitRX/day/eq session and four banks')
 base='rx%d:day%d:eq%d'%(rx,day,eq)
 return base if bank=='original' else base+':virtual-'+bank

def measure(ch,residual,cfg,x,seed,rx,day,eq):
 x=np.asarray(x,dtype=np.complex128)
 if x.shape!=(256,) or not np.isfinite(x).all() or not np.any(abs(x)>0):raise ValueError('One finite nonzero normalized source IQ256 required')
 if cfg.scenario not in d.SCENES or (cfg.fs_hz,cfg.fc_hz,cfg.processing_route,cfg.mode,cfg.equalization_enabled,cfg.reference_snr_db,cfg.boundary)!=(25000000.,2462000000.,'residual','post_sync',False,30.,'reflect'):raise ValueError('Original unchanged practical receiver required')
 outputs={};meta={};reference_trace=None
 for bank in d.BANKS:
  receiver=ch.receiver_for_session(cfg,d.RECEIVER_SEED,session(rx,day,eq,bank));out,m,tr=residual.ResidualChannel(cfg,int(seed),receiver).process(x,return_traces=True)
  if reference_trace is None:reference_trace=tr;reference_meta=m
  if not np.isfinite(out).all() or out.shape!=(256,) or m['channel_equalization_applied'] or not np.array_equal(tr['equalizer_kernel'],np.ones(1,complex)):raise ValueError('Finite full residual with EQoffG1 required')
  for k in ('propagation_kernel','residual_kernel'):
   if not np.array_equal(tr[k],reference_trace[k]):raise ValueError('Session must not change underlying H or numerical boundary')
  for k in ('geometry','state_end','state_counts','state_probabilities','quality_snr_db','residual_scale','receiver_lock_status','propagation_kernel_length','residual_kernel_length','numerical_filter_latency_samples','boundary','noise_boundary'):
   if m[k]!=reference_meta[k]:raise ValueError('Only injected RX realization may change,not geometry/H/quality/state')
  outputs[bank]=out;meta[bank]=m
 original=ch.receiver_for_session(cfg,d.RECEIVER_SEED,session(rx,day,eq,'original'));replay,rm=residual.ResidualChannel(cfg,int(seed),original).process(x)
 if not np.array_equal(replay,outputs['original']):raise ValueError('Same H/RX/noise named stream replay must be exact')
 denom=float(np.mean(abs(outputs['original'])**2));differences={b:dict(relative_IQ_RMS=math.sqrt(float(np.mean(abs(outputs[b]-outputs['original'])**2))/denom),MSE=float(np.mean(abs(outputs[b]-outputs['original'])**2))) for b in d.BANKS}
 return outputs,dict(banks=meta,relative_inputs=differences,original_replay_MSE=0.,native_process_calls=5,same_underlying_H_and_noise_seed=True,processed_noise_not_claimed_equal=True)
