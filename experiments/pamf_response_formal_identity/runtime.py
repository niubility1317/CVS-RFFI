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
            raise ValueError('All four pinned response parent methods required before target access')


def validate_trained(root,arm,parameters,updates):
    trained=read_json(root/arm/'training_complete.json')
    expected=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=updates,L_exposures=126000,U_exposures=1134000,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0,parameters=parameters)
    if any(trained.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed original source-only training ancestry required')


def validate_scored_row(root,arm,*,parameters):
    matches=[v for v in d.ROWS.values() if v['run']==root.name and v['row']==arm and v['parameters']==parameters]
    if root.name==d.BASELINE and arm==d.BASELINE_ROW and parameters==350521:
        info=dict(kind='trained',updates=4940)
    elif len(matches)==1:info=matches[0]
    else:raise ValueError('Registered row and actual trained/reparameterized budget required before IO')
    row=root/arm;done=read_json(row/'complete.json')
    expected=dict(status='ANALYZED',arm=arm,all_predictions_before_truth=True,target_feedback=False)
    if info['kind']=='trained':expected['role']='target'
    else:expected['new_training_updates']=0
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Every fixed row must already finish formal scoring')
    if info['kind']=='trained':validate_trained(root,arm,parameters,info['updates'])
    else:
        original=root.parent/d.r28.COMPLEX_RUN;source_arm=d.r28.SOURCE_ROWS[arm]
        if read_json(original/'launch.json').get('commit')!=d.r28.COMPLEX_COMMIT:raise ValueError('Pinned unchanged native R22 source required')
        validate_trained(original,source_arm,278180,19700)
        expected=dict(source_run=d.r28.COMPLEX_RUN,source_row=source_arm,source_commit=d.r28.COMPLEX_COMMIT,
            inherited_epoch=20,inherited_updates=19700,new_training_updates=0,source_parameters=278180,parameters=299297)
        value=read_json(row/'checkpoint_source.json')
        if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen collapse is not new source training')
        expected=dict(status='FROZEN_RUNTIME_READY',arm=arm,parameters=299297,new_training_updates=0,F_parameter_updates=0,head_fit_updates=0)
        if read_json(row/'runtime_ready.json')!=expected:raise ValueError('Fixed zero-update reparameterized runtime required')
        value=read_json(row/'collapse.json')
        expected=dict(mode='prior_only_linear_pair_collapse',before_parameters=278180,after_parameters=299297,
            factor_modules_replaced=84,unique_kernel_pairs=28,new_training_updates=0,IQ_reads=0,F_reads=0,TX_labels_read=0,
            context_unchanged=True,normalization_unchanged=True,source_model_unchanged=True)
        if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Only registered frozen algebraic conversion allowed')
    folder=row/'target_confirmation/native_map_prior';summary=read_json(folder/'summary.json')
    expected=dict(status='ANALYZED',role='target',count=d.COUNT,original_default_head=True,all_predictions_before_truth=True,
        training_F_access=False,training_TX_labels_read=0,head_fit_updates=0,F_run=d.parent.F_RUN,F_seed=d.parent.F_SEED)
    if any(summary.get(k)!=v for k,v in expected.items()):raise ValueError('Original independent full-seven scorer completion required')
    return folder/'prediction'


def cached_reference(args,ids):
    root=args.runs_root/d.CACHED_RUN
    if read_json(root/'launch.json').get('commit')!=d.CACHED_COMMIT or read_json(root/'complete.json').get('status')!='ANALYZED':
        raise ValueError('Completed fixed R24 analytic cache required')
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
            raise ValueError('All eight full formal rows must finish before analytic target access')
    folders={name:validate_scored_row(args.runs_root/row['run'],row['row'],parameters=row['parameters']) for name,row in d.ROWS.items()}
    baseline=args.runs_root/d.BASELINE
    if read_json(baseline/'launch.json').get('commit')!=d.BASELINE_COMMIT:raise ValueError('Human-designated R46 baseline required')
    folders['r46_350521']=validate_scored_row(baseline,d.BASELINE_ROW,parameters=350521)
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
