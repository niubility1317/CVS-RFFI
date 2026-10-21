import math
from pathlib import Path
import torch
from . import design as d
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_r46_delayed_prune.runtime import load_checkpoint as load_parent

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'R65_four_route_checkpoint_v1'


def build(arm, parent=None):
    if arm == 'stage2':
        from .stage_polyphase import StageDistilled
        return StageDistilled(parent)
    if arm == 'polyphase':
        from .stage_polyphase import PolyphaseRestorer
        return PolyphaseRestorer(parent)
    if arm == 'shared_context':
        from .shared import SharedContextRestorer
        return SharedContextRestorer(parent)
    if arm == 'pcg1':
        from .pcg import PreconditionedRestorer
        return PreconditionedRestorer(parent)
    raise ValueError('One fixed registered route required before any checkpoint IO')


def validate_parent(root):
    if Path(root).name != d.PARENT_RUN:
        raise ValueError('Only user-designated originalR65 run allowed')
    row = Path(root)/d.PARENT_ROW
    done = read_json(row/'training_complete.json')
    expected = dict(status='SOURCE_TRAINED',arm=d.PARENT_ROW,epochs=20,updates=4940,
        L_exposures=126000,U_exposures=1134000,parameters=273765,TX_labels_read=0,
        V_reads=0,target_reads=0,training_F_access=False,F_weights_read=0,F_outputs_read=0,
        optimizer_resets=1,channel_maps_unchanged=True,global_schedule_restarted=False)
    if any(done.get(k) != v for k,v in expected.items()):
        raise ValueError('Complete matching source-only R65 provenance required')
    if read_json(row/'complete.json').get('status') != 'ANALYZED':
        raise ValueError('Reuse complete designated parent artifacts;do not relaunch parent')


def initialize(arm, root, device):
    if arm not in d.ARMS:
        raise ValueError('Registered route required before IO')
    validate_parent(root)
    teacher, payload = load_parent(Path(root)/d.PARENT_ROW/d.KIND/'final.pt','cpu')
    if (payload['arm'],payload['epoch'],payload['updates']) != (d.PARENT_ROW,20,4940):
        raise ValueError('Original R65 source-final required')
    teacher.eval().requires_grad_(False)
    model = build(arm,teacher).requires_grad_(True)
    return model.to(device),teacher.to(device)


def metadata(arm,epoch,updates,initialization,commit):
    return dict(schema=SCHEMA,kind=d.KIND,arm=arm,epoch=epoch,updates=updates,
        config=d.configuration(),parent_run=d.PARENT_RUN,parent_commit=d.PARENT_COMMIT,
        parent_row=d.PARENT_ROW,parent_epoch=20,parent_updates=4940,
        data_contract='original7/63/30',training_TX_labels_read=0,training_F_access=False,
        F_weights_read=0,F_outputs_read=0,V_training=False,target_training=False,
        teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,
        initialization_IQ_examples=0,new_calibration_IQ=0,initialization=initialization,code_commit=commit)


def validate_aliases(model,state=None):
    groups={}
    for name,parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter),[]).append(name)
    if state is not None:
        for names in groups.values():
            if any(not torch.equal(state[names[0]],state[name]) for name in names[1:]):
                raise ValueError('Conflicting shared prior aliases')


def validate_payload(value):
    arm,epoch,updates=value.get('arm'),value.get('epoch'),value.get('updates')
    if arm not in d.ARMS or type(epoch) is not int or not 0<=epoch<=20 or type(updates) is not int or updates!=epoch*247:
        raise ValueError('Complete fixed source epoch or zero-update smoke checkpoint required')
    initial=value.get('initialization')
    if not isinstance(initial,dict) or not isinstance(value.get('code_commit'),str) or len(value['code_commit'])!=40:
        raise ValueError('Initialization record and actual Git commit required')
    expected=metadata(arm,epoch,updates,initial,value['code_commit'])
    if any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('Matching frozen R65 source-only contract required')


def save_checkpoint(path,model,epoch,updates,commit):
    validate_aliases(model)
    value=metadata(model.arm,epoch,updates,model.initialization_report,commit)
    validate_payload(value)
    value['model']=model.state_dict()
    with Path(path).open('xb') as stream:
        torch.save(value,stream)


def load_checkpoint(path,device):
    value=torch.load(path,map_location='cpu',weights_only=False)
    validate_payload(value)
    model=build(value['arm'])
    validate_aliases(model,value['model'])
    model.load_state_dict(value['model'],strict=True)
    model.initialization_report=value['initialization']
    return model.to(device).eval(),value
