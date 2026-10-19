"""Extra complete HQ view: same exogenous propagation, native receiver recomputed."""
from dataclasses import replace,asdict
import re,math
import numpy as np
import torch
FIELDS=('eq_i','rx_i','day_i','sig_i','base_index')
SHARED=('geometry','state_start','state_end','state_counts','state_probabilities','receiver','receiver_location','link_direction','fs_hz','fc_hz','mode','processing_route','input_metrics','orbital_doppler_hz','propagation_kernel_length','residual_kernel_length','residual_kernel_energy','numerical_filter_latency_samples','boundary','noise_boundary','tx_impairment_correction_applied')
def serial(v):
 if isinstance(v,np.generic):return serial(v.item())
 if isinstance(v,np.ndarray):return serial(v.tolist())
 if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [serial(x) for x in v]
 if isinstance(v,complex):return [v.real,v.imag]
 if isinstance(v,float) and not math.isfinite(v):return None
 return v
def tensor_iq(array,device):
 a=np.ascontiguousarray(array,dtype=np.float32)
 return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(device)
def hq_view(primary,raw,meta,first):
 if raw.shape!=(64,2,256) or raw.dtype!=torch.float32 or not bool(torch.isfinite(raw).all()) or len(first)!=64 or any(k not in meta or len(meta[k])!=64 for k in FIELDS):raise ValueError('Exactly64 existing finite sourceL IQ/physical records required')
 vv={k:meta[k].detach().cpu().tolist() if torch.is_tensor(meta[k]) else list(meta[k]) for k in FIELDS};waves=[];out=[];x=np.asarray(raw.detach().cpu().tolist(),dtype=np.float64)
 for i,old in enumerate(first):
  pid=tuple(int(vv[k][i]) for k in FIELDS);session=f'rx{pid[1]}:day{pid[2]}:eq{pid[0]}';namespace=old.get('namespace','');match=re.fullmatch(r'FcrossRX/train/e(\d+)/step(\d+)/L_S',namespace)
  if not match or pid[0]!=1 or pid[1] not in (1,3,4,6,8) or pid[2] not in (1,2,3) or min(pid[3:])<0 or tuple(old.get('sample_id',()))!=pid or old.get('session_id')!=session:raise ValueError('Original sourceL physical ID/session/namespace required')
  ep,step=map(int,match.groups())
  if ep not in range(21,31) or step not in range(4441,6661) or ep!=(step-1)//222+1:raise ValueError('Only original cumulativeE21..30 fullAs view')
  scene=old.get('scene');cfg=primary.configs[scene]
  if scene not in ('practical_low_suburban','practical_mid_urban','practical_low_urban') or cfg.processing_route!='residual' or cfg.mode!='post_sync' or cfg.equalization_enabled or cfg.fs_hz!=25000000. or cfg.fc_hz!=2462000000. or cfg.reference_snr_db!=30. or cfg.boundary!='reflect':raise ValueError('Original full practical configuration required')
  seed=primary.ch.stable_seed(20261016421,namespace,scene,pid)
  if seed!=old.get('seed'):raise ValueError('Same original physical seed required')
  hw=primary.ch.receiver_for_session(cfg,20261016423,session)
  if serial(asdict(hw))!=serial(old['receiver']):raise ValueError('Same full-view session hardware values required')
  hqcfg=replace(cfg,reference_snr_db=36.)
  normalized,_=primary.ch.normalize_input(x[i,0]+1j*x[i,1]);signal,observed=primary.residual.ResidualChannel(hqcfg,seed,hw).process(normalized)
  new=serial(observed)
  if any(new[k]!=serial(old[k]) for k in SHARED):raise ValueError('Exogenous propagation/physical metadata changed')
  if signal.shape!=(256,) or not np.isfinite(signal).all() or new['channel_equalization_applied']:raise ValueError('All finite native HQ outputs required;no dropping')
  waves.append(np.stack((signal.real,signal.imag)));out.append(dict(new,scene=scene,sample_id=pid,session_id=session,namespace=namespace+'/HQ-ref36',physics_namespace=namespace,reference_snr_db=36.,original_reference_snr_db=30.,same_exogenous_physics=True,same_hardware_values=True,receiver_quality_recomputed=True,compensation_recomputed=True,AGC_recomputed=True,full_view_unchanged=True,independent_latent_draw=False,noise_same_standard_stream=True,quality_delta_db=new['quality_snr_db']-old['quality_snr_db']))
 return tensor_iq(waves,raw.device),out
