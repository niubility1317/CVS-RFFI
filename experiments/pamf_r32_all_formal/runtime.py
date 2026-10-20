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
    done=read_json(parent/'complete.json')
    if done.get('status')!='ANALYZED' or done.get('source_rows')!=list(d.ROWS):
        raise ValueError('Wait for the original R32 run to finish')
    records=[read_json(parent/'launch.json'),*read_json(parent/'workers.json')['rows']]
    for record in records:
        proc=Path('/proc')/str(record['pid'])
        try:
            stat=(proc/'stat').read_text();state=stat[stat.rfind(')')+2:].split()[0];cwd=os.readlink(proc/'cwd')
        except FileNotFoundError:continue
        if state!='Z' and cwd==record['cwd']:raise ValueError('Original R32 process is still alive;wait')
    for candidate in d.ROWS:
        row=parent/candidate;validate_training(read_json(row/'training_complete.json'),candidate)
        source=read_json(row/'complete.json')
        required=dict(status='ANALYZED',candidate=candidate,role='source',all_predictions_before_truth=True,training_TX_F_access=False)
        if any(source.get(k)!=v for k,v in required.items()) or not (row/d.KIND/'final.pt').is_file():
            raise ValueError('Both original fixed final checkpoints and completed source diagnostics required')
    # Source accuracies, source_decision and target scores are not consulted.
    return dict(status='FORMAL_MATRIX_FROZEN',rows=list(d.ROWS),kind=d.KIND,parent_run=d.PARENT,parent_commit=d.PARENT_COMMIT,
        policy=d.POLICY,source_performance_gate=False,new_training_updates=0,target_feedback=False,parent_decisions_modified=False)


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
        policy=d.POLICY,source_performance_gate=False,new_training_updates=0,target_feedback=False,parent_decisions_modified=False)
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


def reuse_complete(args):
    """Return an existing same-row result, or None; never decide from performance."""
    audit_target(args)
    row=args.parent_run/args.candidate
    target=row/'target_confirmation'/d.KIND
    flag=row/'target_complete.json'
    if not target.exists() and not flag.exists():return None
    done=read_json(flag)
    required=dict(status='ANALYZED',candidate=args.candidate,role='target',all_predictions_before_truth=True,training_TX_F_access=False)
    if any(done.get(k)!=v for k,v in required.items()):raise ValueError('Existing partial or incompatible target output;preserve without duplicate launch')
    _,ids,_,_=reference(args,'target')
    prediction_arrays(target/'prediction',ids,d.KIND,'target')
    summary=read_json(target/'summary.json')
    expected=dict(status='ANALYZED',kind=d.KIND,role='target',count=168000,F_run=REFERENCE_RUN,F_seed=2026092701,
        original_default_head=True,all_predictions_before_truth=True,training_TX_labels_read=0,training_F_access=False,
        head_fit_updates=0,target_access=True)
    if any(summary.get(k)!=v for k,v in expected.items()) or set(summary['views'])!=set(VIEWS):
        raise ValueError('Existing same-F full seven-view independent score required')
    overall=[r for r in summary['records'] if r['dimension']=='overall' and r['readout']=='restored']
    if len(overall)!=7 or {r['view'] for r in overall}!=set(VIEWS):raise ValueError('All seven existing formal scores required')
    return dict(status='REUSED_COMPLETE',candidate=args.candidate,parent_commit=d.PARENT_COMMIT,
        checkpoint=str(row/d.KIND/'final.pt'),prediction=str(target/'prediction'),summary=str(target/'summary.json'),
        selection_uses_accuracy=False,model_forwards=0,truth_reopened=False)
