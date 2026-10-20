from . import design as d
from experiments.pamf_unrolled_action_fidelity.runtime import ROOT, source_inputs, load_model


def validate_args(args):
    if (args.output.name != d.RUN or args.parent_run.name != d.PARENT
            or args.parent_action.name != d.ACTION or args.reference_run.name != d.REFERENCE):
        raise ValueError('Original source-only RF-response run and parent handles required')
