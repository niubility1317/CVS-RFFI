from . import design as d
from experiments.pamf_unrolled_action_fidelity.runtime import ROOT,load_model,source_inputs as original_inputs
from experiments.pamf_restoration.runner import read_json


def source_inputs(args):
    if (args.covariance_run.name!=d.COV_REFERENCE or args.parent_run.name!=d.PARENT
            or args.parent_action.name!=d.ACTION or args.reference_run.name!=d.REFERENCE):
        raise ValueError('Original source-only R17/R07/R10/R28 handles required')
    launch=read_json(args.covariance_run/'launch.json');done=read_json(args.covariance_run/'complete.json')
    if (launch.get('commit')!=d.COV_REFERENCE_COMMIT or any(done.get(key)!=value for key,value in
            dict(status='ANALYZED',source_only=True,target_reads=0,optimizer_updates=0,source_V_state_updates=0,target_promotion=False).items())):
        raise ValueError('Complete fixed source-only R28 controls provenance differs')
    return original_inputs(args)
