from pathlib import Path
from . import design as d
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference
from experiments.pamf_unrolled_restore.runtime import load_checkpoint
from experiments.pamf_uncertainty_restore.runtime import action_metadata
from experiments.pamf_backfill_restore.runtime import validate_training

ROOT=Path(__file__).resolve().parents[2]


def source_inputs(args):
    if (args.parent_run.name!=d.PARENT or args.parent_action.name!=d.ACTION
            or args.reference_run.name!=d.REFERENCE):raise ValueError('Fixed source-only unrolled parent/action/reference required')
    action_metadata(args.parent_action)
    if read_json(args.reference_run/'launch.json')['commit']!=d.REFERENCE_COMMIT:
        raise ValueError('Original source F50 readout reference provenance differs')
    folder=args.parent_action/'cache/validation';ids,source,index=source_reference(folder,args.source_cache)
    if ids.shape!=(1800,):raise ValueError('Original1800 source panel required')
    return folder,ids,source,index


def load_model(args,device):
    if args.parent_run.name!=d.PARENT or read_json(args.parent_run/'launch.json')['commit']!=d.PARENT_COMMIT:
        raise ValueError('Original fixed R17 source-model provenance required')
    validate_training(read_json(args.parent_run/d.CANDIDATE/'training_complete.json'),d.CANDIDATE)
    model,payload=load_checkpoint(args.parent_run/d.CANDIDATE/'native_map_prior/final.pt',device)
    if (payload['epoch'],payload['updates'],payload['candidate'])!=(20,4940,d.CANDIDATE):
        raise ValueError('Original fixed final E20/4940 unshared model required')
    model.requires_grad_(False).eval()
    return model,dict(run=d.PARENT,commit=d.PARENT_COMMIT,candidate=d.CANDIDATE,
        original_contract='original7/63/30',training_TX_F_reads=0,target_training=False,
        choice='Pre-existing R17 A17-08 source question;not R27 target ranking')
