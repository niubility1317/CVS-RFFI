"""Pure raw-IQ Phase1 predictions; source selection before target, truth last."""
from pathlib import Path
import time
import numpy as np
from experiments.cric_dura_comparison.io import read,write,tensor
from experiments.cric_dura_comparison.evaluation import reject_truth,VIEWS
from .clean_h_objective import METHODS
from .clean_h_source import RUN as SOURCE_RUN,choose_source,REFERENCES,activate
from .restored_source import snapshot,unchanged

BASELINE_RUN='20261026-phase1-PAMF-SIG1-matched-baselines-m6-r01'
TASK_COUNTS=dict(cross_date=30000,cross_rx=126000,cross_both=42000)
SEEDS=(2026092701,2026092702,2026092703)
COMMON={'schema','run_id','row_id','protocol_id','purpose','method','model_seed','source_days','classes','tests','readouts','output_root'}
PREDICT_ONLY={'project','selection_source','source_run_id','source_row_id','source_output','source_contract','frozen_sources'}


def validate(c,scoring=False):
    expected=COMMON|({'prediction_output'} if scoring else PREDICT_ONLY)
    if set(c)!=expected or c['schema']!='pamf_clean_h_raw_eval_v1' or c['protocol_id']!='F0' or c['model_seed'] not in SEEDS or c['source_days']!=[1,2,3] or c['readouts']!=['raw'] or len(c['classes'])!=6:raise ValueError('Declared pure-F F0 evaluation only')
    selected=c['purpose']=='selected_method';reference=c['purpose']=='baseline_cross_date'
    if not (selected or reference) or c['method'] not in (METHODS if selected else ('SIG1-Sup','SIG1-PL')):raise ValueError('Selected clean/H method or existing single-view reference only')
    if set(c['tests'])!=(set(TASK_COUNTS) if selected else {'cross_date'}):raise ValueError('Complete declared physical populations required')
    for task in c['tests'].values():
        if set(task)!=({'views_root','truth','kind'} if scoring else {'views_root'}):raise ValueError('Predictor and truth-side handles must be separate')
    if not scoring:
        reject_truth(c)
        if set(c['frozen_sources'])!={'F','AF','BASE','FOLD'} or c['selection_source']!=c['project']+'/runs/'+SOURCE_RUN+'/complete.json':raise ValueError('No recovery root; fixed source-only selection required')


def selected_method(c):
    selection=read(c['selection_source'])
    if (selection.get('status')!='CLEAN_H_SOURCE_COMPLETE' or selection.get('run_id')!=SOURCE_RUN or selection.get('model_seed')!=2026092701
            or selection.get('U_truth_reads')!=0 or selection.get('target_reads')!=0 or set(selection.get('source_scores',{}))!=set((*REFERENCES,*METHODS))):raise ValueError('Target closed until all source candidates complete')
    _,winner=choose_source(selection['source_scores'])
    if winner is None or selection.get('source_selected')!=winner:raise ValueError('Target closed: fixed source threshold not passed')
    if c['purpose']=='selected_method' and c['method']!=winner:raise ValueError('Only the source-selected method may access target')
    return winner


def source_metadata(src,c,contract):
    cfg=src['config'];reference=c['purpose']=='baseline_cross_date';sup=reference and c['method']=='SIG1-Sup'
    if (src.get('status')!='SOURCE_FROZEN' or src.get('epoch')!=50 or src.get('updates')!=11100 or src.get('target_access') is not False
            or src.get('U_truth_read') is not False or src.get('head_fit_updates')!=0 or cfg['run_id']!=c['source_run_id'] or cfg['row_id']!=c['source_row_id']
            or cfg['model_seed']!=c['model_seed'] or cfg.get('paper_method')!=c['method'] or cfg.get('protocol_id')!='F0' or cfg.get('sig_time_views')!=1
            or cfg.get('default_head_joint') is not True or cfg.get('U_use') is not (not sup) or cfg.get('posthoc_head_fit')
            or contract['classes']!=c['classes'] or read(cfg['source_contract'])!=contract):raise ValueError('Complete matching single-view joint default-head source required')
    if reference and cfg['run_id']!=BASELINE_RUN:raise ValueError('Reuse the already-completed matched reference, no retraining')
    if not reference and (set(cfg.get('frozen_sources',{}))!={'F','AF','BASE','FOLD'} or any(k.startswith('G_') or k in ('source_G','G','restored_roles') for k in cfg)):raise ValueError('Selected source must use only clean/direct H')
    for segment in src['segments']:
        if segment.get('target_access') or segment.get('U_truth_read') or segment.get('head_fit_updates') or segment['config']['U_use'] is not (not sup):raise ValueError('Source ancestry visibility differs')


def array_predictions(reader,array,positions,n,classes,device):
    import torch
    if array.ndim!=3 or array.shape[1:]!=(2,256) or (positions is None and len(array)!=n):raise ValueError('Declared received IQ shape differs')
    if positions is not None and (positions.shape!=(n,) or positions.dtype.kind not in 'iu' or positions.min()<0 or positions.max()>=len(array)):raise ValueError('Received-IQ wrapper positions differ')
    values=[]
    with torch.inference_mode():
        for begin in range(0,n,128):
            ix=slice(begin,begin+128) if positions is None else positions[begin:begin+128]
            x=tensor(np.ascontiguousarray(array[ix],np.float32),device);logits=reader.classify_features(reader.features(x))
            if logits.shape!=(len(x),classes) or not bool(torch.isfinite(logits).all()):raise FloatingPointError('Invalid frozen original-head output')
            values.extend(logits.argmax(1).cpu().tolist())
    if len(values)!=n:raise ValueError('Incomplete physical decisions')
    return np.asarray(values,np.int64)


def predict(c,device='cuda:0'):
    validate(c);winner=selected_method(c)  # Before activation, model load, or any target handle.
    activate(c)
    from experiments.pamf_repl_completion.source import load_reader
    from experiments.cric_dura_comparison.evaluation import handle
    contract=read(c['source_contract']);anchor=read(Path(c['selection_source']).parent/'configs'/(winner+'.json'))
    if contract!=read(anchor['source_contract']):raise ValueError('Checkpoint data contract differs from the selected original F0 source')
    src=read(Path(c['source_output'])/'complete.json');source_metadata(src,c,contract)
    reader,_=load_reader(c['source_output'],contract,device,require_no_unlabeled=c['purpose']=='baseline_cross_date' and c['method']=='SIG1-Sup')
    pool=reader.sequence_pool
    if len(pool.views)!=1 or pool.arm!='SIG_TIME' or pool.modes!=('SIG_TIME',) or tuple(reader.core.id_backbone.cls_head.weight.shape)!=(6,160):raise ValueError('Actual single-view original head required')
    before=snapshot(reader);out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);counts={};started=time.monotonic()
    for task in c['tests']:
        manifest,ids,positions=handle(c['tests'][task]['views_root'],c['classes'])
        if len(ids)!=TASK_COUNTS[task]:raise ValueError('Original F0 physical population differs')
        values={}
        for view in VIEWS:
            path=manifest['clean_ref'] if view=='clean' else str(Path(manifest.get('observations_root',c['tests'][task]['views_root']))/(view+'.npy'))
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            values['raw__'+view]=array_predictions(reader,array,positions,len(ids),6,device)
            write(out/'progress.json',dict(status='PREDICTING',task=task,view=view,physical_packets=len(ids),truth_read=False),exclusive=False)
        folder=out/task;folder.mkdir();np.savez(folder/'predictions.npz',ids=ids,**values)
        counts[task]=dict(count=len(ids),views=list(VIEWS),readouts=['raw'])
    if not unchanged(reader,before):raise ValueError('Frozen F parameters/buffers changed during target prediction')
    write(out/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',run_id=c['run_id'],row_id=c['row_id'],tests=counts,views=list(VIEWS),readouts=['raw'],
        truth_read=False,query_fit=False,parameters_and_buffers_unchanged=True,source_frozen_verified=True,source_selected=winner,seconds=time.monotonic()-started))


def score(c):
    validate(c,scoring=True)
    end=read(Path(c['prediction_output'])/'complete.json')
    if end.get('parameters_and_buffers_unchanged') is not True or end.get('source_frozen_verified') is not True:raise ValueError('Truth closed until frozen-F prediction completes')
    from experiments.cric_dura_comparison import evaluation as baseline
    previous=baseline.tasks;baseline.tasks=lambda _:list(c['tests'])
    try:baseline.score(c)  # Its preflight checks every task/view/ID before the first truth read.
    finally:baseline.tasks=previous
