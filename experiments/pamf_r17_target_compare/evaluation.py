from types import SimpleNamespace
from pathlib import Path
import numpy as np

from . import design as d
from experiments.pamf_r17_task.runtime import load_checkpoint
from experiments.pamf_r17_task.evaluation import score_original_head
from experiments.pamf_uncertainty_restore.evaluation import predict as original_predict
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json
from tools.analyze_pamf_r17_task_results import summary_metrics,compare
from tools.r17_compact_acceptance import target_acceptance


def validate_source(root):
    if root.name!=d.SOURCE_RUN or read_json(root/'launch.json')['commit']!=d.SOURCE_COMMIT:
        raise ValueError('Frozen actual R04 source identity required')
    done=read_json(root/'complete.json')
    if done.get('status')!='ANALYZED' or done.get('source_rows')!=list(d.ARMS) or done.get('target_rows')!=[]:
        raise ValueError('Both completed source-only final checkpoints required')
    for arm in d.ARMS:
        row=root/arm;trained=read_json(row/'training_complete.json')
        expected=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
            teacher_parameter_updates=0,frozen_context_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
            training_F_access=True,F_parameter_updates=0,head_fit_updates=0,parameters=d.PARAMETERS[arm])
        if any(trained.get(k)!=v for k,v in expected.items()) or read_json(row/'complete.json').get('status')!='ANALYZED':
            raise ValueError('Matching full source budget, permissions and original F knowledge required')


def row_args(args):
    if args.arm not in d.ARMS or args.output.name!=d.RUN or args.source_run.name!=d.SOURCE_RUN:
        raise ValueError('Fixed comparison row and roots required before IO')
    return SimpleNamespace(**dict(vars(args),output=args.output/args.arm,kind=d.KIND))


def target_reference(args,role):
    if role!='target' or args.output.name not in d.ARMS or args.output.parent.name!=d.RUN or args.kind!=d.KIND:
        raise ValueError('Only registered fixed comparison target rows allowed before IO')
    # This is an explicit comparison plan, never a forged source qualification.
    if read_json(args.output.parent/'comparison_plan.json')!=d.configuration():
        raise ValueError('Exact registered comparison required before target access')
    manifest=read_json(args.formal_reference/'manifest.json')
    expected=dict(status='FORMAL_REFERENCE_READY',count=168000,views=list(d.VIEWS),original_IQ_bytes_reused=True,TX_truth_reads=0)
    if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Existing formal target handles required')
    ids=np.load(args.formal_reference/'physical_ids.npy',allow_pickle=False)
    if ids.shape!=(168000,) or len(np.unique(ids))!=168000:raise ValueError('Complete fixed target panel required')
    return args.formal_reference,ids,None,None


def predict(args):
    current=row_args(args);validate_source(args.source_run)
    def loader(path,device):
        if path!=current.output/d.KIND/'final.pt':raise ValueError('Registered frozen checkpoint position required')
        model,payload=load_checkpoint(args.source_run/args.arm/d.KIND/'final.pt',device)
        if (payload['arm'],payload['epoch'],payload['updates'])!=(args.arm,20,4940):raise ValueError('Only fixed source final checkpoint allowed')
        return model,payload
    original_predict(current,'target',checkpoint_loader=loader,reference_loader=target_reference)


def score(args):
    current=row_args(args)
    score_original_head(current,'target',reference_loader=target_reference)
    folder=current.output/'target_confirmation'/d.KIND
    student=summary_metrics(read_json(folder/'summary.json'),'target',training_F_access=True)
    original=summary_metrics(read_json(args.teacher_run/'unroll_unshared/target_confirmation'/d.KIND/'summary.json'),'target')
    write_json(current.output/'assessment.json',dict(status='ANALYZED',arm=args.arm,
        user_acceptance=target_acceptance(student,original),target_vs_R17=compare(student,original),
        original_source_waveform_qualified=False,training_updates=0,head_fit_updates=0,target_feedback=False))
    write_json(current.output/'complete.json',dict(status='ANALYZED',arm=args.arm,count=168000,
        all_predictions_before_truth=True,target_feedback=False))
