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
    for run,plan in d.PARENTS.items():
        if read_json(args.runs_root/run/'launch.json').get('commit')!=plan['commit']:
            raise ValueError('Pinned fold/delay/mean and matched-control methods required before target access')
    if read_json(args.runs_root/d.CONTROL_RUN/'launch.json').get('commit')!=d.CONTROL_COMMIT:
        raise ValueError('Pinned predeclared R53 static-graph control required')


def validate_scored_row(root,arm,*,parameters):
    row=root/arm;done=read_json(row/'complete.json')
    expected=dict(status='ANALYZED',arm=arm,role='target',all_predictions_before_truth=True,target_feedback=False)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Already scored fixed full-formal readout required')
    shared=root.name==d.r66.RUN
    training_arm=d.r66.SOURCE_ARM if shared else arm
    trained=read_json(root/training_arm/'training_complete.json')
    expected=dict(status='SOURCE_TRAINED',arm=training_arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0,parameters=parameters)
    if any(trained.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed source-only training ancestry required')
    if shared:
        expected=dict(shared_source_training_paths=1,training_counted_once_across_readouts=True,readouts=list(d.r66.ARMS),EMA_updates=2470,late_mean_snapshots=5,
            new_calibration_IQ=0,new_trainable_parameters=0,extra_training_neural_forwards=0,external_student_checkpoint_reads=0)
        if any(trained.get(k)!=v for k,v in expected.items()):raise ValueError('R66 one source trajectory counted once required')
        complete=read_json(root/'shared_source_complete.json')
        expected=dict(status='SOURCE_TRAINED',source_worker_exited=True,source_training_counted_once=True,readouts=list(d.r66.ARMS))
        if any(complete.get(k)!=v for k,v in expected.items()):raise ValueError('Actual shared source exit required')
        info=read_json(row/'parameter_readout.json')
        expected=dict(readout=arm,rule=d.r66.configuration()['readouts'][arm],shared_source_epochs=20,shared_source_updates=4940,
            ema_updates=2470,late_epoch_snapshots=5,unique_parameters_averaged=714,parameter_aliases=770,buffers_averaged=False,
            teacher_changed=False,extra_neural_forwards=0,new_trainable_parameters=0,TX_reads=0,F_reads=0,V_reads=0,target_reads=0)
        if any(info.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed complete readout metadata required')
    folder=row/'target_confirmation/native_map_prior';summary=read_json(folder/'summary.json')
    expected=dict(status='ANALYZED',role='target',count=d.COUNT,original_default_head=True,all_predictions_before_truth=True,
        training_F_access=False,training_TX_labels_read=0,head_fit_updates=0,F_run=d.parent.F_RUN,F_seed=d.parent.F_SEED)
    if any(summary.get(k)!=v for k,v in expected.items()):raise ValueError('Independent full-seven truth-last scoring required')
    return folder/'prediction'


def cached_reference(args,ids):
    root=args.runs_root/d.CACHED_RUN
    if read_json(root/'launch.json').get('commit')!=d.CACHED_COMMIT or read_json(root/'complete.json').get('status')!='ANALYZED':
        raise ValueError('Completed fixed R56 analytic cache required')
    marker=read_json(root/'feature_complete.json')
    expected=dict(status='FEATURES_COMPLETE',count=d.COUNT,views=list(d.VIEWS),groups=d.GROUPS,
        readouts=d.CACHED_FEATURE_READOUTS,cells=280,TX_labels_read=0,F_access=0,
        model_checkpoint_loads=0,extra_model_predictions=0,new_calibration_samples=0)
    if any(marker.get(k)!=v for k,v in expected.items()):raise ValueError('Unchanged complete label-free cached feature population required')
    reference=read_json(root/'calibration_reference.json')
    expected=dict(run=d.REFERENCE,commit=d.REFERENCE_COMMIT,count=8192,new_calibration_samples=0,new_parameters_fitted=0)
    if any(reference.get(k)!=v for k,v in expected.items()) or reference.get('L_count',0)+reference.get('U_count',0)!=8192:
        raise ValueError('Same original source-only calibration required for cached coordinates')
    if not np.array_equal(np.load(root/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Cached original formal physical IDs must align')
    return root


def inputs(args,*,with_arrays):
    validate_parent(args)
    for run,plan in d.PARENTS.items():
        done=read_json(args.runs_root/run/'complete.json')
        if done.get('status')!='ANALYZED' or done.get('target_rows')!=plan['rows'] or done.get('source_rows')!=plan['rows']:
            raise ValueError('All seven fixed readouts must finish before analytic target access')
    folders={name:validate_scored_row(args.runs_root/row['run'],row['row'],parameters=row['parameters']) for name,row in d.ROWS.items()}
    baseline=args.runs_root/d.BASELINE
    if read_json(baseline/'launch.json').get('commit')!=d.BASELINE_COMMIT:raise ValueError('Human-designated R46 baseline required')
    folders['r46_350521']=validate_scored_row(baseline,d.BASELINE_ROW,parameters=350521)
    folders['r53_prune_decay']=validate_scored_row(args.runs_root/d.CONTROL_RUN,d.CONTROL_ROW,parameters=273765)
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
    cached_reference(args,ids)
    arrays={}
    if with_arrays:
        for view in d.VIEWS:
            paths={name:folders[name]/(view+'.restored.iq.npy') for name in d.NEW_ROWS}
            for name,path in paths.items():
                value=np.load(path,mmap_mode='r',allow_pickle=False)
                if value.shape!=(d.COUNT,2,256) or value.dtype.kind!='f' or not np.isfinite(value).all():
                    raise ValueError('Complete finite saved IQ only;no neural re-prediction')
                arrays[(view,name)]=value
    return ids,arrays,calibration(args)
