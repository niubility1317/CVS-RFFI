from types import SimpleNamespace
import numpy as np
from . import design as d
from .runtime import load_checkpoint
from .context_probe import score_context
from experiments.pamf_uncertainty_restore import evaluation as base
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json
from tools.analyze_pamf_r17_task_results import summary_metrics,compare
from tools.r17_compact_acceptance import target_acceptance
from experiments.pamf_native_formal.runner import assess_goal


def row_args(args):
    if args.arm not in d.ARMS or args.output.name!=d.RUN:raise ValueError('Fixed row/root required')
    return SimpleNamespace(**dict(vars(args),output=args.output/args.arm,kind=d.KIND))


def validate_source_row(row):
    arm=row.name
    if arm not in d.ARMS:raise ValueError('Fixed compact source row required')
    value=read_json(row/'training_complete.json')
    expected=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        parameters=d.PARAMETERS[arm],initialization_IQ_examples=0,teacher_parameter_updates=0,context_updates=4940,prior_updates=4940,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,head_fit_updates=0,V_reads=0,target_reads=0,paired_clean_optimized=True,simulated_physics_anchors=True,physics_target_examples=1260000,action_probe_vectors=2520000)
    if any(value.get(k)!=v for k,v in expected.items()) or read_json(row/'source_complete.json').get('status')!='ANALYZED':
        raise ValueError('Full legal source training and diagnostic predictions required,regardless accuracy')


def reference(args,role):
    if args.output.name not in d.ARMS or args.output.parent.name!=d.RUN or args.kind!=d.KIND:raise ValueError('Registered compact row required before data access')
    if role=='source':return base.reference(args,role)
    if role!='target':raise ValueError('Fixed source/target role required')
    if read_json(args.output.parent/'comparison_plan.json')!=d.configuration():raise ValueError('Fixed all-version comparison plan required')
    validate_source_row(args.output)
    manifest=read_json(args.formal_reference/'manifest.json')
    expected=dict(status='FORMAL_REFERENCE_READY',count=168000,views=list(d.VIEWS),original_IQ_bytes_reused=True,TX_truth_reads=0)
    if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Original unchanged formal reference handles required')
    ids=np.load(args.formal_reference/'physical_ids.npy',allow_pickle=False)
    if ids.shape!=(168000,):raise ValueError('Original fixed formal population required')
    return args.formal_reference,ids,None,None


def predict(args,role):
    current=row_args(args)
    def loader(path,device):
        model,value=load_checkpoint(path,device)
        if (value['arm'],value['epoch'],value['updates'])!=(args.arm,20,4940):raise ValueError('Correct complete final checkpoint required')
        return model,value
    base.predict(current,role,checkpoint_loader=loader,reference_loader=reference)


def score(args,role):
    current=row_args(args);base.score(current,role,reference_loader=reference)
    if role=='source':score_context(args)
    if role=='target':
        folder=current.output/'target_confirmation'/d.KIND
        summary=read_json(folder/'summary.json');student=summary_metrics(summary,'target')
        original=summary_metrics(read_json(args.teacher_run/'unroll_unshared/target_confirmation'/d.KIND/'summary.json'),'target')
        write_json(current.output/'assessment.json',dict(status='ANALYZED',arm=args.arm,user_acceptance=target_acceptance(student,original),
            target_vs_R17=compare(student,original),full_parameters=d.PARAMETERS[args.arm],original_parameters=4665073,target_feedback=False))
        write_json(current.output/'goal_assessment.json',assess_goal(summary))
