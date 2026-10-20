from types import SimpleNamespace
import math
import os
import numpy as np
from experiments.pamf_uncertainty_restore import evaluation as base_eval

from . import design as d
from .runtime import load_checkpoint,validate_teacher_root
from experiments.pamf_uncertainty_restore.evaluation import predict as original_predict,score as original_score
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json


def statistics(summary,role,count,training_F_access=False):
    required=dict(status='ANALYZED',role=role,count=count,F_seed=2026092701,original_default_head=True,
                  all_predictions_before_truth=True,training_F_access=training_F_access,training_TX_labels_read=0,head_fit_updates=0)
    if any(summary.get(k)!=v for k,v in required.items()):
        raise ValueError('Complete same-role original-default-head summary required')
    result={}
    for view in d.VIEWS:
        rows=[v for v in summary['records'] if (v['view'],v['readout'],v['dimension'],v['value'])==(view,'restored','overall','all')]
        if len(rows)!=1 or rows[0]['count']!=count or not math.isfinite(rows[0]['accuracy']):
            raise ValueError('Exactly one complete overall view required')
        result[view]=dict(accuracy=rows[0]['accuracy'],nmse=summary['views'][view]['restored']['waveform']['iq_nmse'])
        if not 0<=result[view]['accuracy']<=100 or not math.isfinite(result[view]['nmse']) or result[view]['nmse']<0:
            raise ValueError('Finite physical accuracy/NMSE required')
    return result


def select_source(candidates,original):
    if set(candidates)!=set(d.ARMS):raise ValueError('Both fixed candidates must complete source evaluation')
    baseline=statistics(original,'source',1800);checks={}
    for arm in d.ARMS:
        value=statistics(candidates[arm],'source',1800,training_F_access=True)
        per_view={v:value[v]['accuracy']>=baseline[v]['accuracy']-d.SOURCE_ALLOWANCE_PP for v in d.VIEWS}
        waveform=sum(value[v]['nmse'] for v in d.VIEWS[1:])<=1.05*sum(baseline[v]['nmse'] for v in d.VIEWS[1:]) and value['clean']['nmse']<=.02
        checks[arm]=dict(qualified=all(per_view.values()) and waveform,per_view_health_accuracy=per_view,
                         per_view_accuracy_nonregression={v:value[v]['accuracy']>=baseline[v]['accuracy'] for v in d.VIEWS},
                         waveform=waveform,delta_pp={v:value[v]['accuracy']-baseline[v]['accuracy'] for v in d.VIEWS})
    eligible=sorted((a for a in d.ARMS if checks[a]['qualified']),key=lambda a:(d.PARAMETERS[a],a))
    selected=eligible[0] if eligible else None
    return dict(status='SOURCE_DECISION_FROZEN',qualified=bool(selected),selected=selected,
                target_rows=[selected] if selected else [],checks=checks,source_eligible=eligible,
                source_screening_allowance_pp=d.SOURCE_ALLOWANCE_PP,target_acceptance_allowance_pp=0.,source_uses_V_labels=True,V_state_updates=0,target_reads_before_decision=0,target_feedback=False,
                rule=d.configuration()['source_selection'])


def row_args(args):
    return SimpleNamespace(**dict(vars(args),output=args.output/args.arm,kind=d.KIND))


def prepare_target(args):
    value=read_json(args.output/'source_decision.json')
    if (value.get('status')!='SOURCE_DECISION_FROZEN' or not value.get('qualified') or value.get('selected')!=args.arm
            or args.arm not in d.ARMS or value.get('target_rows')!=[args.arm] or value.get('target_reads_before_decision')!=0
            or value.get('target_feedback') is not False):
        raise ValueError('Only one source-selected compact candidate may access target')
    write_json(args.output/args.arm/'source_decision.json',dict(value,target_candidates=[d.KIND]))


def predict(args,role):
    if role=='target':prepare_target(args)
    elif role!='source' or args.arm not in d.ARMS:raise ValueError('Registered source candidate required')
    def loader(path,device):
        model,payload=load_checkpoint(path,device)
        if payload['arm']!=args.arm:raise ValueError('Wrong compact arm')
        return model,payload
    original_predict(row_args(args),role,checkpoint_loader=loader)


def score(args,role):
    score_original_head(row_args(args),role)
    if role=='target':
        validate_teacher_root(args.teacher_run)
        folder=args.output/args.arm/'target_confirmation'/d.KIND
        current=statistics(read_json(folder/'summary.json'),'target',168000,training_F_access=True)
        baseline=statistics(read_json(args.teacher_run/'unroll_unshared/target_confirmation'/d.KIND/'summary.json'),'target',168000)
        checks={v:current[v]['accuracy']>=baseline[v]['accuracy'] for v in d.VIEWS}
        write_json(args.output/args.arm/'compression_assessment.json',dict(status='ANALYZED',
            target_accuracy_nonregression=all(checks.values()),per_view=checks,
            delta_pp={v:current[v]['accuracy']-baseline[v]['accuracy'] for v in d.VIEWS},
            full_parameters=d.PARAMETERS[args.arm],original_parameters=4665073,
            target_feedback=False,goal_complete=False))


def score_original_head(args,role):
    """Same original truth-last scorer, with honest F-knowledge metadata."""
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU scorer required')
    folder,ids,source,index=base_eval.reference(args,role)
    row=base_eval.row_folder(args,role)
    predictions=base_eval.prediction_arrays(row/'prediction',ids,args.kind,role)
    if role=='source':
        refs=base_eval.prediction_arrays(args.output/'reference_controls/prediction',ids,'reference_controls','source')
        baseline={view:refs[view]['identity']['logits'] for view in d.VIEWS}
        clean_features=refs['clean']['identity']['features']
        labels=np.load(args.source_cache/'V/labels.npy',allow_pickle=False)[index]
        rx,day=source.rx[index],source.day[index]
    else:
        baseline,clean_features=base_eval.target_baseline(args,ids)
        truth=read_json(args.target_truth)
        labels=np.asarray([truth[str(pid)]['label'] for pid in ids],np.int64)
        rx=np.asarray([str(truth[str(pid)]['receiver']) for pid in ids])
        day=np.asarray([str(truth[str(pid)].get('day','unavailable')) for pid in ids])
    clean=np.load(folder/'clean.npy',mmap_mode='r',allow_pickle=False)
    clean_unit=clean_features/np.maximum(np.linalg.norm(clean_features,axis=1,keepdims=True),1e-8)
    records=[];views={}
    for view in d.VIEWS:
        views[view]={};before=baseline[view].argmax(1)
        for name,arrays in predictions[view].items():
            metrics=base_eval.waveform(arrays['iq'],clean);decision=arrays['logits'].argmax(1)
            groups=[('overall','all',np.ones(len(ids),bool))]
            for dimension,values in (('TX',labels),('RX',rx),('day',day)):
                groups.extend((dimension,str(value),values==value) for value in np.unique(values))
            for dimension,value,mask in groups:
                records.append(dict(view=view,readout=name,dimension=dimension,value=value,
                    **base_eval.classification_statistics(labels[mask],decision[mask],before[mask]),
                    iq_nmse=float(metrics['iq_nmse'][mask].mean())))
            unit=arrays['features']/np.maximum(np.linalg.norm(arrays['features'],axis=1,keepdims=True),1e-8)
            views[view][name]=dict(waveform={key:float(value.mean()) for key,value in metrics.items()},
                paired_clean_feature_cosine_distance=float((1-(unit*clean_unit).sum(1)).mean()),
                paired_clean_vs_other_TX=base_eval.capped_pair_margin(unit,clean_unit,labels,512))
    write_json(row/'summary.json',dict(status='ANALYZED',kind=args.kind,role=role,count=len(ids),views=views,records=records,
        F_run=d.F_RUN,F_seed=d.F_SEED,original_default_head=True,all_predictions_before_truth=True,
        training_TX_labels_read=0,training_F_access=True,F_in_training='frozen original source-supervised teacher;0parameterupdates',
        head_fit_updates=0,target_access=role=='target',goal_complete=False))
