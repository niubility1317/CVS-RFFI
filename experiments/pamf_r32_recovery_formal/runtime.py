import os
from pathlib import Path
import numpy as np
from . import design as d
from experiments.pamf_response_restore.runtime import load_checkpoint as original_load
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_uncertainty_restore.evaluation import VIEWS,prediction_arrays
from experiments.pamf_fcrossrx_readout.frozen import REFERENCE_RUN

ROOT=Path(__file__).resolve().parents[2]


def validate_training(value,candidate):
    if candidate not in d.ROWS:raise ValueError('Fixed R32 row required')
    expected=dict(status='SOURCE_TRAINED',candidate=candidate,epochs=20,updates=4940,
        L_exposures=126000,U_exposures=1134000,L_waveform_forwards=378000,U_waveform_forwards=3402000,
        TX_labels_read=0,F_outputs_read=0,F_weights_read=0,target_reads=0,frozen_context_updates=0,
        prior_initialization='scratch',inherited_prior_epochs=0,response_constraint_active=candidate=='observable_response')
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Original complete TX/F-free R32 training required')


def parent_ready(args):
    parent=args.parent_run
    if parent.name!=d.PARENT or read_json(parent/'launch.json')['commit']!=d.PARENT_COMMIT:
        raise ValueError('Original pinned R32 lineage required')
    failure=read_json(parent/'failure.json')
    if failure.get('status')!='FAILED_TECHNICAL' or (parent/'complete.json').exists():
        raise ValueError('Only the documented failed source-scorer parent is recovered')
    records=[read_json(parent/'launch.json'),*read_json(parent/'workers.json')['rows']]
    for record in records:
        proc=Path('/proc')/str(record['pid'])
        try:
            stat=(proc/'stat').read_text();state=stat[stat.rfind(')')+2:].split()[0];cwd=os.readlink(proc/'cwd')
        except FileNotFoundError:continue
        if state!='Z' and cwd==record['cwd']:raise ValueError('Original R32 process still alive;wait')
    for candidate in d.ROWS:
        row=parent/candidate;validate_training(read_json(row/'training_complete.json'),candidate)
        if not (row/d.KIND/'final.pt').is_file():raise ValueError('Complete fixed final checkpoint required')
        log=(parent/'logs'/('train-'+candidate+'.log')).read_text(errors='replace')
        if 'ValueError: not enough values to unpack (expected 4, got 3)' not in log:
            raise ValueError('Recovery is limited to the diagnosed source-reference arity failure')
        prediction=read_json(row/d.KIND/'prediction/complete.json')
        response=read_json(row/'source_response/complete.json')
        expected=dict(status='PREDICTIONS_COMPLETE',role='source',kind=d.KIND,count=1800,views=list(VIEWS),TX_truth_reads=0,state_updates=0,head_fit_updates=0)
        if any(prediction.get(k)!=v for k,v in expected.items()):raise ValueError('Complete original source predictions required')
        expected=dict(status='SOURCE_RESPONSE_COMPLETE',run=d.PARENT,candidate=candidate,count=1800,cells=12,TX_truth_reads=0,F_access=0,target_reads=0,optimizer_updates=0,state_updates=0)
        if any(response.get(k)!=v for k,v in expected.items()):raise ValueError('Complete original source response predictions required')
        if (row/'target_confirmation').exists() or (row/'target_complete.json').exists():
            raise ValueError('This failed-parent recovery expects zero prior target inference')
    return dict(status='FORMAL_MATRIX_FROZEN',rows=list(d.ROWS),kind=d.KIND,parent_run=d.PARENT,parent_commit=d.PARENT_COMMIT,
        policy=d.POLICY,source_performance_gate=False,new_training_updates=0,target_feedback=False,parent_decisions_modified=False,
        parent_state='SOURCE_SCORER_FAILED_TRAINING_SAVED')


def load_final(path,device):
    candidate=Path(path).parents[1].name
    if candidate not in d.ROWS:raise ValueError('Registered row checkpoint path required')
    model,value=original_load(path,device)
    if value.get('candidate')!=candidate or value.get('epoch')!=20 or value.get('updates')!=4940 or value.get('kind')!=d.KIND:
        raise ValueError('Exact complete R32 row checkpoint required')
    return model.requires_grad_(False).eval(),value


def audit_target(args):
    if args.output.name!=args.candidate or args.candidate not in d.ROWS or args.kind!=d.KIND:
        raise ValueError('Only the fixed two-row formal matrix may access target')
    root=args.output.parent
    if read_json(root/'resolved_config.json')!=d.configuration(args.expected_commit):raise ValueError('Registered configuration required')
    matrix=read_json(root/'formal_matrix.json')
    expected=dict(status='FORMAL_MATRIX_FROZEN',rows=list(d.ROWS),kind=d.KIND,parent_run=d.PARENT,parent_commit=d.PARENT_COMMIT,
        policy=d.POLICY,source_performance_gate=False,new_training_updates=0,target_feedback=False,parent_decisions_modified=False,parent_state='SOURCE_SCORER_FAILED_TRAINING_SAVED')
    if matrix!=expected:raise ValueError('Explicit all-version formal authorization required')


def reference(args,role):
    if role!='target':raise ValueError('This fixed evaluation has no source fitting or prediction')
    audit_target(args)
    manifest=read_json(args.formal_reference/'manifest.json')
    required=dict(status='FORMAL_REFERENCE_READY',count=168000,views=list(VIEWS),original_IQ_bytes_reused=True,TX_truth_reads=0)
    if any(manifest.get(k)!=v for k,v in required.items()):raise ValueError('Original formal reference handles required')
    ids=np.load(args.formal_reference/'physical_ids.npy',allow_pickle=False)
    if ids.shape!=(168000,):raise ValueError('Original formal physical panel required')
    return args.formal_reference,ids,None,None
