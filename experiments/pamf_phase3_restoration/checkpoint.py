"""Strict standalone TF state; no DURA/checkpoint inheritance."""
from pathlib import Path
import re
import torch
from .model import scratch_model, architecture
from .cache import CAPSULE, SPLIT, SCHEMA as CACHE_SCHEMA

RUNS={
    'IQ':'20261101-phase3-P3Restore-TF-IQ-E30-s2026092701-r01',
    'F0':'20261101-phase3-P3Restore-TF-F0-E30-s2026092701-r01',
}
SCHEMA='p3_restore_TF_checkpoint_v1'


def training_config(arm):
    if arm not in RUNS:raise ValueError('Fixed IQ/F0 matched arm required')
    return dict(run_id=RUNS[arm],arm=arm,model_seed=2026092701,data_seed=2026110101,
        architecture=architecture(),epochs=30,updates=7410,batch=256,
        optimizer='AdamW',learning_rate=.001,weight_decay=1e-4,
        warmup_updates=247,minimum_learning_rate=2e-5,global_gradient_clip=1.,
        clean_input_probability=.15,objective='IQNMSE+.05mean3STFT_relativeL1',
        feature_guide='0.1fullF0fused160cos/per-recordIQgradientcap.5' if arm=='F0' else None,
        training_F_access=arm=='F0',training_roles={'L_s':6300,'U_s':56700},
        cache_schema=CACHE_SCHEMA,capsule_id=CAPSULE,split_id=SPLIT,
        initialization='scratch',ancestor_checkpoints=[],training_stages=1,
        TX_labels_read=0,OE_reads=0,V_reads=0,target_reads=0,
        checkpoint_selection='final30_only',default_DURA_changed=False)


def save_model(path,model,arm,commit,epoch,updates):
    if re.fullmatch('[0-9a-f]{40}',commit) is None:raise ValueError('Actual code Git commit required')
    if (epoch,updates) not in [(0,0),(30,7410)]:raise ValueError('Only E0 smoke or fixed E30 required')
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite G state')
    payload=dict(schema=SCHEMA,config=training_config(arm),code_commit=commit,
                 epoch=epoch,updates=updates,state_dict=state)
    with Path(path).open('xb') as stream:torch.save(payload,stream)


def load_model(path,device,*,arm,commit,epoch=30):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if (set(value)!= {'schema','config','code_commit','epoch','updates','state_dict'}
            or value['schema']!=SCHEMA or value['config']!=training_config(arm)
            or value['code_commit']!=commit or value['epoch']!=epoch
            or value['updates']!=(0 if epoch==0 else 7410) or epoch not in (0,30)):
        raise ValueError('Complete original-source standalone TF checkpoint required')
    model=scratch_model(2026092701)
    state=value['state_dict']
    if (not isinstance(state,dict) or set(state)!=set(model.state_dict())
            or any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32
                   or v.shape!=model.state_dict()[k].shape or not torch.isfinite(v).all() for k,v in state.items())):
        raise ValueError('Exact finite FP32 standalone TF state required')
    model.load_state_dict(state,strict=True)
    return model.to(device).eval().requires_grad_(False)
