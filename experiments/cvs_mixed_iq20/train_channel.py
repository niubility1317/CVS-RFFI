"""Source IQ views, named independent streams; full original practical engine."""
from dataclasses import asdict
import numpy as np
import torch
from .augmentation import apply_response,session_response
from . import train_design as d


def engine():
    from experiments.cvs_phase1_overlay.model import native_modules
    native_modules()
    import leo_practical.channel as ch
    import leo_practical.residual as residual
    return ch,residual


def configurations():
    ch,_=engine()
    return {s:ch.Config(fs_hz=25_000_000.,scenario=s,processing_route='residual',mode='post_sync',
                       equalization_enabled=False,reference_snr_db=30.,boundary='reflect') for s in d.SCENES}


def audit():
    ch,r=engine()
    return dict(channel_module_file=ch.__file__,residual_module_file=r.__file__,
                actual_configs={s:asdict(c) for s,c in configurations().items()},
                EQ_off_G_equals_one=True,formal_test_changed=False,unknown_TX_inverse=False,
                normalization='native source RMS then simulator unit RMS; no simulator output RMS')


def tensor_iq(a,device):
    a=np.ascontiguousarray(a,dtype=np.float32)
    return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(device)


def metadata(extra,count):
    meta=extra[1] if isinstance(extra,(tuple,list)) and len(extra)>1 else extra
    def values(k):
        v=meta[k]
        return v.detach().cpu().tolist() if torch.is_tensor(v) else list(v)
    result={k:values(k) for k in ('rx_i','day_i','eq_i','base_index')}
    if any(len(v)!=count for v in result.values()):raise ValueError('Source physical metadata length mismatch')
    if not set(result['rx_i'])<={1,3,4,6,8} or not set(result['day_i'])<={1,2,3} or set(result['eq_i'])!={1}:
        raise ValueError('Only original equalized1 source RX/day allowed')
    result['ids']=list(meta['physical_id']) if 'physical_id' in meta else ['source:'+str(v) for v in result['base_index']]
    return result


def generalization(x,meta,epoch):
    coefficients=[session_response(20261015031,stream='train',epoch=epoch,rx=int(rx),day=int(day),eq=int(eq))
                  for rx,day,eq in zip(meta['rx_i'],meta['day_i'],meta['eq_i'])]
    out,detail=apply_response(x,coefficients)
    power=x.square().sum(1).mean(1)
    error=((out.square().sum(1).mean(1)-power).abs()/power.clamp_min(1e-12)).max()
    if float(error)>2e-6:raise ValueError('Generalization response lost packet power')
    return out,dict(max_relative_power_error=float(error),waveform_MSE=float((out-x).square().mean()),
                    session_coefficients=[[a.real,a.imag] for a in coefficients])


def practical(x,meta,epoch,step,*,source_scene=None):
    ch,residual=engine();configs=configurations();raw=np.asarray(x.detach().cpu().tolist(),dtype=np.float64)
    out=[];records=[]
    for i,pid in enumerate(meta['ids']):
        namespace='mixedIQ/source/fixed' if source_scene else f'mixedIQ/train/epoch{epoch}/step{step}/L_S'
        if source_scene:
            scene=source_scene
        else:
            rng=np.random.default_rng(ch.stable_seed(20261015032,namespace,'scene',pid))
            scene=str(rng.choice(list(d.SCENES),p=[1/30,1/30,.3,1/30,.3,.3]))
        cfg=configs[scene]
        seed=ch.stable_seed(20261015033 if source_scene else 20261015032,namespace,scene,pid)
        session=f"rx{meta['rx_i'][i]}:day{meta['day_i'][i]}:eq{meta['eq_i'][i]}"
        hardware=ch.receiver_for_session(cfg,2026092701,session)
        record,_=ch.normalize_input(raw[i,0]+1j*raw[i,1])
        value,m=residual.ResidualChannel(cfg,seed,hardware).process(record)
        m.update(sample_id=pid,session_id=session,namespace=namespace)
        out.append(np.stack((value.real,value.imag)));records.append(m)
    return tensor_iq(np.stack(out),x.device),records
