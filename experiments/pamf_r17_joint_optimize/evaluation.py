from types import SimpleNamespace
import numpy as np
from . import design as d
from .runtime import load_checkpoint
from experiments.pamf_r17_task.evaluation import score_original_head
from experiments.pamf_uncertainty_restore import evaluation as base
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json
from tools.analyze_pamf_r17_task_results import summary_metrics,compare
from tools.r17_compact_acceptance import target_acceptance


def row_args(args):
    if args.arm not in d.ARMS or args.output.name!=d.RUN:raise ValueError('Registered fixed row/root required before IO')
    return SimpleNamespace(**dict(vars(args),output=args.output/args.arm,kind=d.KIND))


def validate_source_row(row):
    arm=row.name
    if arm not in d.ARMS:raise ValueError('Registered source row required')
    value=read_json(row/'training_complete.json')
    required=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        parameters=d.PARAMETERS[arm],initialization_IQ_examples=0,teacher_parameter_updates=0,context_updates=4940,context_trainable=True,
        TX_labels_read=0,V_reads=0,target_reads=0,training_F_access=True,F_parameter_updates=0,head_fit_updates=0)
    if any(value.get(k)!=v for k,v in required.items()) or read_json(row/'source_complete.json').get('status')!='ANALYZED':
        raise ValueError('Full source budget and completed diagnostic predictions required')


def reference(args,role):
    if args.output.name not in d.ARMS or args.output.parent.name!=d.RUN or args.kind!=d.KIND:
        raise ValueError('Registered compact row required before data access')
    if role=='source':return base.reference(args,role)
    if role!='target':raise ValueError('Registered source/target role required')
    plan=read_json(args.output.parent/'comparison_plan.json')
    required=dict(schema='R17_joint_optimize_v1',run=d.RUN,arms=list(d.ARMS),views=list(d.VIEWS),
        target_count_per_view=168000,target_acceptance_allowance_pp=1.,target_feedback=False)
    if any(plan.get(k)!=v for k,v in required.items()):raise ValueError('Prospective fixed two-row comparison required')
    validate_source_row(args.output)
    manifest=read_json(args.formal_reference/'manifest.json')
    expected=dict(status='FORMAL_REFERENCE_READY',count=168000,views=list(d.VIEWS),original_IQ_bytes_reused=True,TX_truth_reads=0)
    if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Matching formal target handles required')
    ids=np.load(args.formal_reference/'physical_ids.npy',allow_pickle=False)
    if ids.shape!=(168000,) or len(np.unique(ids))!=168000:raise ValueError('Complete fixed target physical IDs required')
    return args.formal_reference,ids,None,None


def predict(args,role):
    current=row_args(args)
    def loader(path,device):
        if path!=current.output/d.KIND/'final.pt':raise ValueError('Registered frozen final position required')
        model,payload=load_checkpoint(path,device)
        if (payload['arm'],payload['epoch'],payload['updates'])!=(args.arm,20,4940):
            raise ValueError('Full final source checkpoint required before prediction')
        return model,payload
    base.predict(current,role,checkpoint_loader=loader,reference_loader=reference)


def score(args,role):
    current=row_args(args);score_original_head(current,role,reference_loader=reference)
    if role=='target':
        folder=current.output/'target_confirmation'/d.KIND
        student=summary_metrics(read_json(folder/'summary.json'),'target',training_F_access=True)
        original=summary_metrics(read_json(args.teacher_run/'unroll_unshared/target_confirmation'/d.KIND/'summary.json'),'target')
        write_json(current.output/'assessment.json',dict(status='ANALYZED',arm=args.arm,
            user_acceptance=target_acceptance(student,original),target_vs_R17=compare(student,original),
            full_parameters=d.PARAMETERS[args.arm],original_parameters=4665073,target_feedback=False))
