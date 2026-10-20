from pathlib import Path
import torch
from . import design as d
from experiments.pamf_r44_pruned_refine import runtime as parent
from experiments.pamf_r46_conditional_factor import runtime as factor
from experiments.pamf_r46_complex_fidelity import runtime as complex_runtime
from experiments.pamf_restoration.runner import read_json


def source_root(args, arm):
    if arm not in d.ARMS:
        raise ValueError('Fixed diagnostic arm required before checkpoint IO')
    key, name = (('compact_parent', d.COMPACT_PARENT_RUN) if arm == d.ARMS[0] else
                 ('factor_run', d.FACTOR_RUN) if arm == d.ARMS[1] else ('complex_run', d.COMPLEX_RUN))
    root = Path(getattr(args, key))
    if root.name != name:
        raise ValueError('Fixed source-final checkpoint root required before IO')
    return root


def validate_training(value):
    required = dict(status='SOURCE_TRAINED', epochs=20, updates=19700, parameters=278180,
        L_exposures=126000, U_exposures=1134000, training_F_access=False,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0)
    if any(value.get(key) != expected for key,expected in required.items()):
        raise ValueError('Complete matching source-only training budget required')


def validate_sources(args):
    roots = {arm:source_root(args, arm) for arm in d.ARMS}
    complex_runtime.validate_parent_root(roots[d.REFERENCE_ARM])
    for arm in d.ARMS[1:]:
        root = roots[arm]
        expected = d.FACTOR_COMMIT if arm == d.ARMS[1] else d.COMPLEX_COMMIT
        if read_json(root/'launch.json').get('commit') != expected:
            raise ValueError('Fixed source method commit required')
        validate_training(read_json(root/d.SOURCE_ROWS[arm]/'training_complete.json'))


def load_model(args, arm):
    root = source_root(args, arm)
    loader = (parent.load_checkpoint if arm == d.ARMS[0] else
              factor.load_checkpoint if arm == d.ARMS[1] else complex_runtime.load_checkpoint)
    model, payload = loader(root/d.SOURCE_ROWS[arm]/d.KIND/'final.pt', 'cpu')
    if (payload['arm'],payload['epoch'],payload['updates']) != (d.SOURCE_ROWS[arm],*d.SOURCE_BUDGET[arm]):
        raise ValueError('Fixed source-final checkpoint required')
    if sum(p.numel() for p in model.parameters()) != d.PARAMETERS[arm]:
        raise ValueError('Full restorer parameter count differs')
    return model.eval().requires_grad_(False), payload


def check_frozen(model, snapshot):
    state = model.state_dict()
    if (set(state) != set(snapshot) or any(not torch.equal(value, snapshot[key]) for key,value in state.items())
            or any(p.grad is not None or p.requires_grad for p in model.parameters())
            or any(module.training for module in model.modules())):
        raise ValueError('Frozen source checkpoint state or evaluation mode changed')
