from pathlib import Path
import numpy as np
from . import design as d
from experiments.pamf_restoration.runner import read_json

ROOT=Path(__file__).resolve().parents[2]


def validate_paths(args):
    if (args.output.name!=d.RUN or args.output.parent!=args.runs_root
            or args.formal_reference!=args.runs_root/d.FORMAL/'reference'
            or args.target_truth!=args.runs_root/d.TRUTH_RUN/'data/truth.json'):
        raise ValueError('Registered existing formal inputs and new own output required before IO')


def calibration(args):
    root=args.runs_root/d.REFERENCE
    if read_json(root/'launch.json').get('commit')!=d.REFERENCE_COMMIT:raise ValueError('Original R34 source-only calibration required')
    value=read_json(root/'calibration.json')
    expected=dict(count=8192,label_reads=0,F_access=False,V_fit=False,role='source_LU_only',V_fit_updates=0,target_reads=0)
    if any(value.get(k)!=v for k,v in expected.items()) or value.get('L_count',0)+value.get('U_count',0)!=8192:
        raise ValueError('Unchanged label-free LU-only moments required')
    return value


def validate_parent(args):
    validate_paths(args)
    if read_json(args.runs_root/d.PARENT/'launch.json').get('commit')!=d.PARENT_COMMIT:
        raise ValueError('Pinned R44 parent required before target access')


def validate_scored_row(root,arm,*,new):
    row=root/arm
    done=read_json(row/'complete.json')
    if any(done.get(k)!=v for k,v in dict(status='ANALYZED',arm=arm,role='target',all_predictions_before_truth=True,target_feedback=False).items()):
        raise ValueError('Already scored fixed target row required')
    trained=read_json(row/'training_complete.json')
    expected=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0,
        parameters=d.parent.PARAMETERS[arm] if new else 794529)
    if any(trained.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed F/TX-free training ancestry required')
    folder=row/'target_confirmation/native_map_prior';summary=read_json(folder/'summary.json')
    expected=dict(status='ANALYZED',role='target',count=d.COUNT,original_default_head=True,
        all_predictions_before_truth=True,training_F_access=False,training_TX_labels_read=0,head_fit_updates=0,
        F_run=d.parent.F_RUN,F_seed=d.parent.F_SEED)
    if any(summary.get(k)!=v for k,v in expected.items()):raise ValueError('Original independent full-seven scorer completion required')
    return folder/'prediction'


def inputs(args,*,with_arrays):
    validate_parent(args);parent=args.runs_root/d.PARENT
    done=read_json(parent/'complete.json')
    if (done.get('status')!='ANALYZED' or done.get('target_rows')!=list(d.NEW_ROWS)
            or done.get('source_rows')!=list(d.NEW_ROWS)):
        raise ValueError('Both full formal rows must finish before analytic target access')
    folders={arm:validate_scored_row(parent,arm,new=True) for arm in d.NEW_ROWS}
    baseline=args.runs_root/d.BASELINE
    if read_json(baseline/'launch.json').get('commit')!=d.BASELINE_COMMIT:raise ValueError('User-designated R38 baseline required')
    folders['r38_794529']=validate_scored_row(baseline,'dense_half',new=False)
    reference=read_json(args.formal_reference/'manifest.json')
    if any(reference.get(k)!=v for k,v in dict(status='FORMAL_REFERENCE_READY',count=d.COUNT,views=list(d.VIEWS),original_IQ_bytes_reused=True,TX_truth_reads=0).items()):
        raise ValueError('Unchanged complete original formal reference required')
    ids=np.load(args.formal_reference/'physical_ids.npy',allow_pickle=False)
    if ids.shape!=(d.COUNT,) or len(np.unique(ids))!=d.COUNT:raise ValueError('Original unique formal physical IDs required')
    for folder in folders.values():
        marker=read_json(folder/'complete.json')
        expected=dict(status='PREDICTIONS_COMPLETE',kind='native_map_prior',role='target',count=d.COUNT,
            views=list(d.VIEWS),readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,
            F_run=d.parent.F_RUN,F_seed=d.parent.F_SEED)
        if any(marker.get(k)!=v for k,v in expected.items()) or not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):
            raise ValueError('All candidates full-seven matched original predictions required')
    arrays={}
    if with_arrays:
        for view in d.VIEWS:
            paths={'raw':args.formal_reference/(view+'.npy'),**{name:folder/(view+'.restored.iq.npy') for name,folder in folders.items()}}
            for name,path in paths.items():
                value=np.load(path,mmap_mode='r',allow_pickle=False)
                if value.shape!=(d.COUNT,2,256) or value.dtype.kind!='f' or not np.isfinite(value).all():
                    raise ValueError('Complete finite saved IQ only;no neural re-prediction')
                arrays[(view,name)]=value
    return ids,arrays,calibration(args)
