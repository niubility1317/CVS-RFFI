from pathlib import Path
from . import design as d
from experiments.pamf_analytic_restore.runtime import initialize
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference

ROOT=Path(__file__).resolve().parents[2]


def source_inputs(args):
    if args.parent_run.name!=d.PARENT or args.parent_action.name!=d.ACTION:
        raise ValueError('Only fixed source parent and action cache allowed')
    handle=read_json(args.parent_action/'cache/complete.json')
    if (handle.get('status')!='ACTION_SOURCE_CACHE_COMPLETE' or handle.get('train_count')!=63000
            or handle.get('source_V_count')!=1800 or handle.get('TX_labels_read')!=0 or handle.get('target_reads')!=0):
        raise ValueError('Original source-only cache handles required')
    folder=args.parent_action/'cache/validation'
    ids,source,index=source_reference(folder,args.source_cache)
    if ids.shape!=(1800,):raise ValueError('Original1800 source panel required')
    return folder,ids,source,index


def load_model(args,device):
    model,parent,_=initialize(args.parent_run,device)
    model.requires_grad_(False).eval()
    return model,parent
