"""Epoch-boundary continuation of the fixed, source-only pure response rows.

Legacy v1 pure rows isolate every stochastic update with its saved ticket seed.
Their outer RNG does not determine updates; accepting them is explicit and is
limited to this audited route. Controllers, grid samplers and joint DR rows are
intentionally unsupported rather than silently losing their state.
"""
from copy import deepcopy
from pathlib import Path
import torch
from cvsrffi.game_tracking.state import RNGState
from cvsrffi.game_tracking.runtime import plain

IGNORED_ARGS={'output_dir','device','wisig_pkl','game_resume','xuc_resume',
              'xuc_allow_legacy_ticket_resume','xuc_pause_request','xuc_stop_after_epoch'}

def capture_resume_state(*,actions,head_total,elapsed,counter=None):
    return dict(version=1,rng=RNGState.capture(),actions=dict(actions),head_total=head_total,
                elapsed_seconds=elapsed,forward_counts=counter.snapshot() if counter else None,
                boundary='after_epoch_validation',randomness='isolated_ticket_seed')

def comparable_args(args):
    return plain({k:v for k,v in args.items() if k not in IGNORED_ARGS})

def validate_checkpoint(payload,args,row,source,steps,*,allow_legacy=False):
    if not row.get('pure_game') or row.get('joint',{}).get('native_dr') is not False:
        raise ValueError('RESUME_UNSUPPORTED: only pure-game rows')
    if row.get('ticket_curriculum_enabled') or row.get('control')!='off' or row.get('carrier')=='v2_grid':
        raise ValueError('RESUME_UNSUPPORTED: stateful controller or grid sampler')
    required={'model','ema','optimizer','prototype','args','row','epoch','step','source_info',
              'solver','tickets','daot_rc4','initialization','from_scratch','target_contact'}
    if required-set(payload):raise ValueError('RESUME_MISSING_STATE: '+str(sorted(required-set(payload))))
    if payload.get('schema')!='adv3b02_xuc_v1' or payload['row']!=row:
        raise ValueError('RESUME_ROW_MISMATCH')
    if payload['target_contact'] is not False or payload['initialization']!='scratch_only' or payload['from_scratch'] is not True:
        raise ValueError('CHECKPOINT_TARGET_CONTAMINATED_OR_PROVENANCE_UNVERIFIED')
    if comparable_args(payload['args'])!=comparable_args(vars(args)):
        raise ValueError('RESUME_CONFIG_MISMATCH')
    if plain(payload['source_info'])!=plain(source.info):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    completed=payload['step']
    if type(completed) is not int or completed!=payload['epoch']*steps or not 0<completed<=args.epochs*steps:
        raise ValueError('RESUME_NOT_EPOCH_BOUNDARY')
    if completed==args.epochs*steps:raise ValueError('RESUME_ALREADY_COMPLETE')
    if payload['solver']['steps']!=completed or payload['daot_rc4']['commits']!=completed:
        raise ValueError('RESUME_COMMIT_CLOCK_MISMATCH')
    tickets=payload['tickets']
    if tickets['sampler'] is not None or tickets['consumed']!=set(range(completed)):
        raise ValueError('RESUME_TICKET_COVERAGE_MISMATCH')
    pending_ids=[t.ticket_id for t in tickets['pending']]
    if sorted(pending_ids)!=list(range(completed,(tickets['next_epoch']-1)*steps)):
        raise ValueError('RESUME_NEXT_TICKET_MISMATCH')
    if not payload['epoch']<tickets['next_epoch']<=args.epochs+1:
        raise ValueError('RESUME_NEXT_EPOCH_MISMATCH')
    if any(payload.get(k) is not None for k in ('cstar','legacy_controller','legacy_curriculum')):
        raise ValueError('RESUME_UNSUPPORTED_CONTROLLER_STATE')
    dr=payload['daot_rc4']
    if dr['calibration'] or dr['route_history']:
        raise ValueError('RESUME_PURE_GAME_CONTAINS_RC4_STATE')
    meta=payload.get('resume_state')
    if meta is None:
        if not allow_legacy:raise ValueError('RESUME_MISSING_RNG: explicit legacy ticket replay required')
    elif meta.get('version')!=1 or meta.get('boundary')!='after_epoch_validation' or not isinstance(meta.get('rng'),RNGState):
        raise ValueError('RESUME_INVALID_STATE')
    return completed

def to_device(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,dict):return {k:to_device(v,device) for k,v in value.items()}
    if isinstance(value,list):return [to_device(v,device) for v in value]
    if isinstance(value,tuple):return tuple(to_device(v,device) for v in value)
    return deepcopy(value)

def restore_checkpoint(payload,model,ema,optimizer,proto,solver,stream,dr,counter=None):
    model.load_state_dict(payload['model'],strict=True)
    if ema is None or payload['ema'] is None:raise ValueError('RESUME_MISSING_EMA')
    ema.load_state_dict(payload['ema'],strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    device=next(model.parameters()).device
    vars(proto).clear();vars(proto).update(to_device(payload['prototype'],device))
    solver.mode=payload['solver']['mode'];solver.load_state_dict(payload['solver'])
    for key in ('next_epoch','pending','consumed','window','window_consumed'):
        setattr(stream,key,deepcopy(payload['tickets'][key]))
    saved=payload['daot_rc4'];dr.scale.load_state_dict(saved['scale'])
    for key in ('calibration','calibration_epoch','commits','route_history'):
        setattr(dr,key,deepcopy(saved[key]))
    meta=payload.get('resume_state')
    if meta:
        if counter is not None and meta['forward_counts'] is not None:counter.counts.update(meta['forward_counts'])
        meta['rng'].restore()
        return deepcopy(meta)
    # Legacy outside-update RNG is deliberately not fabricated. Every stochastic
    # update is reseeded in isolated_rng(ticket.seed); source reads are center crops.
    return dict(actions={'NORMAL':payload['step'],'CATCHUP':0,'CORRECT':0},head_total=0,
                elapsed_seconds=0.,legacy_outer_rng_not_restored=True)

def read_checkpoint(path):
    path=Path(path).resolve()
    if path.name not in ('latest_ssdg.pth','final_ssdg.pth'):
        raise ValueError('RESUME_EXPECTS_NATIVE_EPOCH_CHECKPOINT')
    return torch.load(path,map_location='cpu',weights_only=False)

def validate_resume_payload(payload,row,source_info,args=None,*,steps=222,allow_legacy=False):
    """Read-only validation entry point for a stable snapshot before process stop."""
    from types import SimpleNamespace
    args=args if args is not None else SimpleNamespace(**payload['args'])
    return validate_checkpoint(payload,args,row,SimpleNamespace(info=source_info),steps,allow_legacy=allow_legacy)
