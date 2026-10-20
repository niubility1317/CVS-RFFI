from pathlib import Path
import torch
from . import design as d
from experiments.pamf_r46_conditional_factor import runtime as factor
from experiments.pamf_r44_pruned_refine import runtime as parent
from experiments.pamf_restoration.runner import read_json

ROOT = Path(__file__).resolve().parents[2]


def source_root(args, arm):
    if arm not in d.ARMS:
        raise ValueError('Registered graph arm required before checkpoint IO')
    return args.compact_parent if arm == d.ARMS[0] else args.factor_run


def validate_sources(args):
    if args.teacher_run.name != d.PARENT_RUN or args.parent_action.name != d.ACTION_RUN:
        raise ValueError('Original R17 and learned-action source handles required before IO')
    factor.validate_parent_root(args.compact_parent)
    if args.factor_run.name != d.FACTOR_RUN or read_json(args.factor_run/'launch.json')['commit'] != d.FACTOR_COMMIT:
        raise ValueError('Fixed legal R18 source run required')
    if read_json(args.factor_run/'comparison_plan.json') != d.factor.configuration():
        raise ValueError('Original fixed R18 matrix required')
    for arm in d.ARMS[1:]:
        value = read_json(args.factor_run/d.SOURCE_ROWS[arm]/'training_complete.json')
        required = dict(status='SOURCE_TRAINED', epochs=20, updates=19700, parameters=278180,
            L_exposures=126000, U_exposures=1134000, training_F_access=False, TX_labels_read=0,
            F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0)
        if any(value.get(k) != v for k,v in required.items()):
            raise ValueError('Complete source-only R18 budget required')


def source_checkpoint(args, arm):
    return source_root(args, arm)/d.SOURCE_ROWS[arm]/d.KIND/'final.pt'


def load_model(args, arm, device):
    root = source_root(args, arm)
    if root.name != (d.COMPACT_PARENT_RUN if arm == d.ARMS[0] else d.FACTOR_RUN):
        raise ValueError('Fixed source root required')
    loader = parent.load_checkpoint if arm == d.ARMS[0] else factor.load_checkpoint
    model, payload = loader(source_checkpoint(args, arm), device)
    if (payload['arm'],payload['epoch'],payload['updates']) != (d.SOURCE_ROWS[arm],*d.SOURCE_BUDGET[arm]):
        raise ValueError('Fixed source-final checkpoint required')
    if sum(p.numel() for p in model.parameters()) != d.PARAMETERS[arm]:
        raise ValueError('Complete restorer parameter count differs')
    return model.eval().requires_grad_(False), payload
