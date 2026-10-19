from functools import partial
from types import SimpleNamespace
import torch
from . import design as d,runtime
from .calibration import validate_metadata
from .descriptors import AnalyticViews,paired_loss
from experiments.pamf_fullresolution_restore.training import train as original_train,learning_rate
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json


def objective_gradients(output,metrics,analytic):
    def norm(value):
        g=torch.autograd.grad(value.mean(),output,retain_graph=True)[0]
        if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite actual objective gradient')
        return float(g.norm())
    result=dict(IQ=norm(metrics['iq_nmse']))
    if analytic:
        result['weighted_analytic']=norm(.01*metrics['analytic_loss'])
        for key in ('mirror','spectral','reference'):result[key]=norm(.01*metrics[key+'_loss']/3)
        result['analytic_to_IQ_ratio']=result['weighted_analytic']/result['IQ'] if result['IQ']>0 else None
    return result


def train(args):
    if args.row not in d.ROWS or args.parent_action.name!=d.ACTION:raise ValueError('Original fullLU scratch analytic row required')
    meta=read_json(args.output/'analytic_scale.json');scales=validate_metadata(meta);analytic=args.row=='analytic_views'
    extractor=AnalyticViews().to('cuda:0') if analytic else None
    measured=False
    def loss(output,clean):
        nonlocal measured
        paired,metrics=paired_loss(output,clean,scales if analytic else None,analytic=analytic,extractor=extractor)
        if not measured:
            gradients=objective_gradients(output,metrics,analytic)
            write_json(args.output/args.row/'objective_gradient_first_batch.json',dict(status='SOURCE_TRAINING_GRADIENTS_RECORDED',
                actual_calibrated_scales=True,count=len(output),optimizer_updates_before=0,gradient_accumulated=False,
                objective_output_gradients=gradients,analytic_branch_executed=analytic,TX_labels_read=0,F_access=False,target_access=False))
            measured=True
        return paired,metrics
    methods=SimpleNamespace(initialize=runtime.initialize,load_checkpoint=runtime.load_checkpoint,
        save_checkpoint=partial(runtime.save_checkpoint,calibration=meta))
    original_train(args,configuration=d.configuration,model_runtime=methods,loss_fn=loss)
    write_json(args.output/args.row/'objective_complete.json',dict(status='SOURCE_OBJECTIVE_TRAINED',candidate=args.row,
        analytic_branch_executed=analytic,analytic_weight=.01 if analytic else 0.,calibration_count=8192,
        calibration_shared_across_rows=True,calibration_optimizer_updates=0,training_TX_F_access=False,target_access=False))
