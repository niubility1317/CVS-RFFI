from pathlib import Path
from . import design as d
from experiments.pamf_r65_four_route.runtime import load_checkpoint
from experiments.pamf_restoration.runner import read_json

ROOT=Path(__file__).resolve().parents[2]


def validate_source(args):
    if args.parent_run.name!=d.PARENT_RUN or args.parent_action.name!=d.ACTION_RUN:
        raise ValueError('User-designated two-stage/source handles required before IO')
    if read_json(args.parent_run/'launch.json')['commit']!=d.PARENT_COMMIT:
        raise ValueError('Original two-stage source method commit differs')
    row=args.parent_run/d.PARENT_ROW
    done=read_json(row/'training_complete.json')
    required=dict(status='SOURCE_TRAINED',arm=d.PARENT_ROW,epochs=20,updates=4940,parameters=d.PARAMETERS,
        training_F_access=False,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0)
    if any(done.get(k)!=v for k,v in required.items()):
        raise ValueError('Complete matching source-only two-stage ancestry required')
    if read_json(row/'complete.json')['status']!='ANALYZED':
        raise ValueError('Original frozen two-stage row must be complete; other routes need not finish')


def checkpoint(args):
    if args.parent_run.name!=d.PARENT_RUN:
        raise ValueError('Only user-designated two-stage checkpoint allowed')
    return args.parent_run/d.PARENT_ROW/d.KIND/'final.pt'


def load_model(args,device):
    model,payload=load_checkpoint(checkpoint(args),device)
    if (payload['arm'],payload['epoch'],payload['updates'],payload['code_commit'])!=(d.PARENT_ROW,20,4940,d.PARENT_COMMIT):
        raise ValueError('Original two-stage source-final checkpoint required')
    if sum(p.numel() for p in model.parameters())!=d.PARAMETERS or len(model.prior)!=2:
        raise ValueError('Original complete two-stage structure required')
    return model.eval().requires_grad_(False),payload
