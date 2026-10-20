from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import design as d
from experiments.pamf_analytic_identity_probe.runner import inputs as reference_inputs,load_features as reference_features
from experiments.pamf_restoration.runner import read_json

ROOT=Path(__file__).resolve().parents[2]


def validate_training(value,row):
    if row not in d.NEW_ROWS:raise ValueError('Registered compact row required')
    required=dict(status='SOURCE_TRAINED',arm=row,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        TX_labels_read=0,F_outputs_read=0,F_weights_read=0,training_F_access=False,V_reads=0,target_reads=0,
        teacher_parameter_updates=0,context_updates=4940,prior_updates=4940,paired_clean_optimized=True,
        parameters=d.PARAMETERS[row],initialization_IQ_examples=0,simulated_physics_anchors=True,physics_target_examples=1260000,action_probe_vectors=2520000)
    if any(value.get(k)!=v for k,v in required.items()):raise ValueError('Complete fixed TX/F-free compact training required')


def validate_source_row(value,row):
    required=dict(status='ANALYZED',arm=row,role='source',all_predictions_before_truth=True,source_selection=False,diagnostic_only=True,target_feedback=False)
    if any(value.get(k)!=v for k,v in required.items()):raise ValueError('Completed matching source-only compact row required')


def sources(args,*,with_arrays):
    parent=args.runs_root/d.PARENT
    if read_json(parent/'launch.json').get('commit')!=d.PARENT_COMMIT:raise ValueError('Pinned physically anchored compact source run required')
    # No target metrics,source rank or target-completion state is consumed.
    for name in d.NEW_ROWS:
        validate_training(read_json(parent/name/'training_complete.json'),name)
        validate_source_row(read_json(parent/name/'source_complete.json'),name)
    previous=args.runs_root/d.REFERENCE
    if read_json(previous/'launch.json').get('commit')!=d.REFERENCE_COMMIT:raise ValueError('Original R34 feature provenance required')
    complete=read_json(previous/'complete.json')
    if complete.get('status')!='ANALYZED' or any(complete.get(k)!=0 for k in ('new_neural_training_updates','head_fit_updates','F_weights_read','F_embeddings_logits_read','target_reads')):
        raise ValueError('Complete F-independent source R34 reference required')
    ids,source,index,_=reference_inputs(args)
    if not np.array_equal(np.load(previous/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Original R34 panel differs')
    result={};arrays={}
    if with_arrays:
        result=reference_features(SimpleNamespace(output=previous),ids)
    for name in d.NEW_ROWS:
        folder=parent/name/'native_map_prior/prediction';header=read_json(folder/'complete.json')
        expected=dict(status='PREDICTIONS_COMPLETE',kind='native_map_prior',role='source',count=1800,views=list(d.VIEWS),
            readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0)
        if any(header.get(k)!=v for k,v in expected.items()) or not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):
            raise ValueError('All matching seven-view compact source IQ predictions required')
        if with_arrays:
            for view in d.VIEWS:
                value=np.load(folder/(view+'.restored.iq.npy'),mmap_mode='r',allow_pickle=False)
                if value.shape!=(1800,2,256) or not np.isfinite(value).all():raise ValueError('Complete finite compact IQ required')
                arrays[(view,name)]=value
    calibration=read_json(previous/'calibration.json')
    required=dict(count=8192,label_reads=0,F_access=False,V_fit=False,role='source_LU_only',V_fit_updates=0,target_reads=0)
    if any(calibration.get(k)!=v for k,v in required.items()) or calibration['L_count']+calibration['U_count']!=8192:
        raise ValueError('Original label-free LU-only R34 moments required')
    return ids,source,index,result,arrays,calibration
