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
    if args.arm not in d.ARMS or args.output.name!=d.RUN:raise ValueError('Fixed route/root required')
    return SimpleNamespace(**dict(vars(args),output=args.output/args.arm,kind=d.KIND))


def reference(args,role):
    if args.output.name not in d.ARMS or args.output.parent.name!=d.RUN or args.kind!=d.KIND:
        raise ValueError('Fixed registered row before data IO required')
    done=read_json(args.output/'training_complete.json')
    required=dict(status='SOURCE_TRAINED',arm=args.output.name,epochs=20,updates=4940,
        L_exposures=126000,U_exposures=1134000,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,
        training_F_access=False,V_reads=0,target_reads=0,teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0)
    if any(done.get(k)!=v for k,v in required.items()):raise ValueError('Complete legal source training before evaluation required')
    if read_json(args.output.parent/'comparison_plan.json')!=d.configuration():raise ValueError('Frozen four-route plan required')
    if role=='source':return base.reference(args,role)
    if role!='target':raise ValueError('Fixed source/target role required')
    if read_json(args.output/'source_complete.json').get('status')!='ANALYZED':raise ValueError('Source artifacts must close before formal prediction')
    manifest=read_json(args.formal_reference/'manifest.json')
    expected=dict(status='FORMAL_REFERENCE_READY',count=168000,views=list(d.VIEWS),original_IQ_bytes_reused=True,TX_truth_reads=0)
    if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Original fixed formal data handles required')
    ids=np.load(args.formal_reference/'physical_ids.npy',allow_pickle=False)
    if ids.shape!=(168000,) or len(np.unique(ids))!=168000:raise ValueError('Complete original physical population required')
    return args.formal_reference,ids,None,None


def predict(args,role):
    current=row_args(args)
    def loader(path,device):
        if path!=current.output/d.KIND/'final.pt':raise ValueError('Only this fixed final checkpoint may predict')
        model,value=load_checkpoint(path,device)
        if (value['arm'],value['epoch'],value['updates'],value['code_commit'])!=(args.arm,20,4940,args.expected_commit):
            raise ValueError('Complete fixed source-final checkpoint required')
        return model,value
    base.predict(current,role,checkpoint_loader=loader,reference_loader=reference,checkpoint_budget=(20,4940))


def score(args,role):
    current=row_args(args)
    base.score(current,role,reference_loader=reference)
    if role=='target':
        student=summary_metrics(read_json(current.output/'target_confirmation'/d.KIND/'summary.json'),'target',training_F_access=False)
        original=summary_metrics(read_json(args.teacher_run/'unroll_unshared/target_confirmation'/d.KIND/'summary.json'),'target')
        parent=summary_metrics(read_json(args.parent_run/d.PARENT_ROW/'target_confirmation'/d.KIND/'summary.json'),'target',training_F_access=False)
        write_json(current.output/'assessment.json',dict(status='ANALYZED',arm=args.arm,
            original_R17_acceptance=target_acceptance(student,original),R65_degradation_diagnostic=target_acceptance(student,parent),
            target_vs_R65=compare(student,parent),target_vs_R17=compare(student,original),
            target_feedback=False,goal_complete=False))
