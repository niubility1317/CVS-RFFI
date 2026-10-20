import math
from pathlib import Path
import torch
from . import design as d
from .collapse import collapse_prior
from experiments.pamf_r46_complex_fidelity import runtime as source_runtime
from experiments.pamf_r46_complex_fidelity.models import CompactSuccessor
from experiments.pamf_restoration.runner import read_json

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='R46_prior_collapse_checkpoint_v1'


def source_root(args,arm):
    if arm not in d.ARMS:raise ValueError('Registered frozen collapse arm required before IO')
    if args.complex_run.name!=d.COMPLEX_RUN:raise ValueError('Fixed R22 source root required before IO')
    return args.complex_run


def validate_sources(args):
    if args.teacher_run.name!=d.PARENT_RUN or args.parent_action.name!=d.ACTION_RUN:
        raise ValueError('Original source and reference handles required before IO')
    source_runtime.validate_parent_root(args.compact_parent)
    root=source_root(args,d.ARMS[0])
    if read_json(root/'launch.json')['commit']!=d.COMPLEX_COMMIT or read_json(root/'comparison_plan.json')!=d.source.configuration():
        raise ValueError('Fixed source-only R22 configuration required')
    for arm in d.ARMS:
        value=read_json(root/d.SOURCE_ROWS[arm]/'training_complete.json')
        expected=dict(status='SOURCE_TRAINED',arm=d.SOURCE_ROWS[arm],epochs=20,updates=19700,parameters=278180,
            L_exposures=126000,U_exposures=1134000,training_F_access=False,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0)
        if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Complete F-free source-final R22 required')


def source_checkpoint(args,arm):
    return source_root(args,arm)/d.SOURCE_ROWS[arm]/d.KIND/'final.pt'


def metadata(arm,payload,audit):
    if arm not in d.ARMS:raise ValueError('Registered collapse arm required')
    return dict(schema=SCHEMA,config=d.configuration(),arm=arm,kind=d.KIND,parameters=d.PARAMETERS[arm],
        epoch=20,updates=19700,source_run=d.COMPLEX_RUN,source_commit=d.COMPLEX_COMMIT,source_row=d.SOURCE_ROWS[arm],
        source_payload={k:v for k,v in payload.items() if k not in ('model','optimizer')},collapse=audit,
        new_training_updates=0,new_training_IQ_examples=0,F_training_access=False,TX_training_labels_read=0,
        F_parameter_updates=0,head_fit_updates=0,optimizer=None)


def validate_payload(value):
    arm=value.get('arm')
    if arm not in d.ARMS:raise ValueError('Registered collapse payload required')
    payload=value.get('source_payload');audit=value.get('collapse')
    if not isinstance(payload,dict) or not isinstance(audit,dict):raise ValueError('Source metadata and collapse audit required')
    source_runtime.validate_payload(payload)
    if (payload['arm'],payload['epoch'],payload['updates'])!=(d.SOURCE_ROWS[arm],20,19700):raise ValueError('Fixed source-final inherited budget required')
    expected=metadata(arm,payload,audit)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen collapse contract differs')
    required=dict(mode='prior_only_linear_pair_collapse',before_parameters=278180,after_parameters=299297,
        factor_modules_replaced=84,unique_kernel_pairs=28,context_unchanged=True,normalization_unchanged=True,
        source_model_unchanged=True,new_training_updates=0,IQ_reads=0,F_reads=0,TX_labels_read=0)
    if any(audit.get(k)!=v for k,v in required.items()) or not math.isfinite(audit.get('seconds',-1)) or audit.get('seconds',-1)<0:
        raise ValueError('Prior-only frozen conversion required')


def validate_aliases(model,state=None):
    if sum(p.numel() for p in model.parameters())!=299297:raise ValueError('Complete expanded parameter count differs')
    aliases={}
    for name,p in model.named_parameters(remove_duplicate=False):aliases.setdefault(id(p),[]).append(name)
    if state is not None:
        for names in aliases.values():
            if len(names)>1 and any(not torch.equal(state[names[0]],state[name]) for name in names[1:]):
                raise ValueError('Conflicting shared expanded weight aliases')
    return model


def convert_source(args,arm):
    source,payload=source_runtime.load_checkpoint(source_checkpoint(args,arm),'cpu')
    if (payload['arm'],payload['epoch'],payload['updates'])!=(d.SOURCE_ROWS[arm],20,19700):raise ValueError('Fixed R22 final required')
    source.eval().requires_grad_(False);model,audit=collapse_prior(source)
    validate_payload(metadata(arm,payload,audit));return validate_aliases(model),source,payload,audit


def save_checkpoint(path,model,arm,payload,audit):
    value=metadata(arm,payload,audit);validate_payload(value);validate_aliases(model)
    value['model']=model.state_dict()
    with Path(path).open('xb') as stream:torch.save(value,stream)


def load_checkpoint(path,device):
    value=torch.load(path,map_location='cpu',weights_only=False);validate_payload(value)
    template=CompactSuccessor(d.SOURCE_ROWS[value['arm']]).eval().requires_grad_(False)
    model,_=collapse_prior(template);validate_aliases(model,value['model']);model.load_state_dict(value['model'],strict=True)
    return validate_aliases(model.to(device)).eval().requires_grad_(False),value
