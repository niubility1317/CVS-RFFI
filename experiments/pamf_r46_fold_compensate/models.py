import torch
from . import design as d
from .compensation import grouped_links,make_links,folded_weight
from experiments.pamf_r46_compact_refine.models import CompactSuccessor as Existing273765,compact_from_parent as weight_only
from experiments.pamf_r46_multipoint_fast.models import objective as original_objective,spectral_error

def CompactSuccessor(arm):
    if arm not in d.ARMS:raise ValueError('Registered channel reconstruction arm required')
    model=Existing273765('prune_refine');model.arm=arm;return model

def weight_initialization(parent):
    model=weight_only(parent,'prune_refine')
    maps={name:[torch.tensor(v,dtype=torch.int64) for v in levels] for name,levels in model.initialization_report['channel_maps'].items()}
    if any(any(not torch.equal(a,b) for a,b in zip(maps['prior0'],maps['prior'+str(i)])) for i in (1,2)):raise ValueError('Original weight-only maps must preserve shared prior semantics')
    return model,maps

@torch.no_grad()
def compact_from_parent(parent,arm,calibration):
    if arm not in d.ARMS:raise ValueError('Registered reconstruction arm required')
    from .calibration import validate_calibration
    validate_calibration(calibration)
    model,maps=weight_initialization(parent);groups=grouped_links(make_links(parent,model,maps));expected={}
    for group in groups:
        first=group[0]
        if len(first.inputs)==first.source.weight.shape[1]:continue
        expected[first.name]=group
    if set(expected)!=set(calibration['projections']):raise ValueError('Exact source-calibrated weight groups required')
    for name,group in expected.items():
        item=calibration['projections'][name];first=group[0]
        if item['members']!=[v.name for v in group] or item['inputs']!=first.inputs.tolist() or item['outputs']!=first.outputs.tolist():raise ValueError('Source projection differs from fixed weight maps or aliases')
        projection=torch.tensor(item['projection'],dtype=torch.float64)
        first.target.weight.copy_(folded_weight(first,projection,d.FOLD_STRENGTH[arm]))
    model.arm=arm;model.requires_grad_(True)
    model.initialization_report=dict(mode=d.MODE[arm],group_count=4,IQ_examples=256,source_unique_IQ=128,calibration_L_exposures=128,calibration_U_exposures=128,
        TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True,exact_function_equivalence=False,
        shared_prior_maps=True,calibration_seed=d.CAL_SEED,ridge_fraction=d.RIDGE_FRACTION,fold_strength=d.FOLD_STRENGTH[arm],
        folded_unique_weights=len(expected),folded_module_uses=sum(len(g) for g in expected.values()),no_intercept=True,
        channel_maps={name:[v.tolist() for v in levels] for name,levels in maps.items()})
    return model

def objective(student,teacher,clean,epoch,arm):
    if arm not in d.ARMS:raise ValueError('Registered channel reconstruction arm required')
    return original_objective(student,teacher,clean,epoch)
