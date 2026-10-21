from types import SimpleNamespace
import numpy as np
from . import design as d
from .runtime import load_checkpoint
from experiments.pamf_uncertainty_restore import evaluation as base
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json
from tools.analyze_pamf_r17_task_results import summary_metrics,compare
from tools.r17_compact_acceptance import target_acceptance


def row_args(args):
    if args.arm not in d.ARMS or args.output.name!=d.RUN:raise ValueError('Fixed successor row/root required')
    return SimpleNamespace(**dict(vars(args),output=args.output/args.arm,kind=d.KIND))


def validate_source_row(row):
    arm=row.name
    if arm not in d.ARMS:raise ValueError('Registered successor source row required')
    value=read_json(row/'training_complete.json')
    expected=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,parameters=d.PARAMETERS_BY_ARM[arm],
        initialization_IQ_examples=0,teacher_parameter_updates=0,context_updates=4940,prior_updates=4940,cumulative_prior_epochs=100,
        cumulative_coarse_epochs=100,cumulative_action_epochs=100,TX_labels_read=0,V_reads=0,target_reads=0,training_F_access=d.TASK[arm],
        F_weights_read=0,F_outputs_read=0,F_parameter_updates=0,head_fit_updates=0,paired_clean_optimized=True,focus_fraction=d.FRACTION[arm],focus_mix=d.MIX,
        hard_exposures=int(1260000*d.FRACTION[arm]),extra_training_G_forwards=0,new_calibration_IQ=0,new_trainable_parameters=0,
        persistent_hardness_state=False,teacher_term_reweighted=False)
    if any(value.get(k)!=v for k,v in expected.items()) or read_json(row/'source_complete.json').get('status')!='ANALYZED':raise ValueError('Complete legal successor source budget required')


def reference(args,role):
    if args.output.name not in d.ARMS or args.output.parent.name!=d.RUN or args.kind!=d.KIND:raise ValueError('Registered successor row before IO required')
    if role=='source':return base.reference(args,role)
    if role!='target':raise ValueError('Fixed source/target role required')
    if read_json(args.output.parent/'comparison_plan.json')!=d.runtime_configuration():raise ValueError('Fixed reconstruction-tail comparison required')
    validate_source_row(args.output)
    manifest=read_json(args.formal_reference/'manifest.json')
    expected=dict(status='FORMAL_REFERENCE_READY',count=168000,views=list(d.VIEWS),original_IQ_bytes_reused=True,TX_truth_reads=0)
    if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Original target reference handles required')
    ids=np.load(args.formal_reference/'physical_ids.npy',allow_pickle=False)
    if ids.shape!=(168000,) or len(np.unique(ids))!=168000:raise ValueError('Same complete physical population required')
    return args.formal_reference,ids,None,None


def predict(args,role):
    current=row_args(args)
    def loader(path,device):
        if path!=current.output/d.KIND/'final.pt':raise ValueError('Fixed final path required')
        model,value=load_checkpoint(path,device)
        if (value['arm'],value['epoch'],value['updates'])!=(args.arm,20,4940):raise ValueError('Fixed complete successor final required')
        return model,value
    base.predict(current,role,checkpoint_loader=loader,reference_loader=reference,checkpoint_budget=(d.EPOCHS,d.UPDATES))


def validate_mean_control(root):
    if read_json(root/'launch.json').get('commit')!=d.CONTROL['commit']:raise ValueError('Pinned mean-objective control required')
    row=root/d.CONTROL['row']
    if (row/'failure.json').exists():raise ValueError('Mean control failed;preserve outputs')
    if not (row/'complete.json').is_file():
        if (root/'failure.json').exists():raise ValueError('Control ended without final row;preserve outputs')
        return False
    expected=dict(status='ANALYZED',arm=d.CONTROL['row'],role='target',all_predictions_before_truth=True,target_feedback=False)
    if any(read_json(row/'complete.json').get(k)!=v for k,v in expected.items()):raise ValueError('Complete fixed final control row required')
    trained=read_json(root/'source_path/training_complete.json')
    expected=dict(status='SOURCE_TRAINED',epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,parameters=273765,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0,shared_source_training_paths=1,training_counted_once_across_readouts=True)
    if any(trained.get(k)!=v for k,v in expected.items()):raise ValueError('Legal single-path source control required')
    return True


def score(args,role):
    current=row_args(args)
    control_root=args.output.parent/d.CONTROL['run']
    if role=='target' and not validate_mean_control(control_root):raise ValueError('Wait for final control before entering truth scorer')
    base.score(current,role,reference_loader=reference)
    if role=='target':
        student=summary_metrics(read_json(current.output/'target_confirmation'/d.KIND/'summary.json'),'target',training_F_access=False)
        original=summary_metrics(read_json(args.teacher_run/'unroll_unshared/target_confirmation'/d.KIND/'summary.json'),'target')
        compact=summary_metrics(read_json(args.compact_parent/d.COMPACT_PARENT_ROW/'target_confirmation'/d.KIND/'summary.json'),'target')
        r44=summary_metrics(read_json(args.compact_parent.parent/d.parent.COMPACT_PARENT_RUN/'group_pruned/target_confirmation'/d.KIND/'summary.json'),'target')
        control=summary_metrics(read_json(control_root/d.CONTROL['row']/'target_confirmation'/d.KIND/'summary.json'),'target')
        acceptance=target_acceptance(student,compact);original_acceptance=target_acceptance(student,original)
        write_json(current.output/'assessment.json',dict(status='ANALYZED',arm=args.arm,user_acceptance=acceptance,original_R17_acceptance=original_acceptance,
            accepted_both=acceptance['accepted'] and original_acceptance['accepted'],target_vs_R46=compare(student,compact),target_vs_R17=compare(student,original),target_vs_R44=compare(student,r44),
            target_vs_mean_control=compare(student,control),mean_control=d.CONTROL,full_parameters=d.PARAMETERS_BY_ARM[args.arm],parent_parameters=350521,original_parameters=4665073,target_feedback=False))
