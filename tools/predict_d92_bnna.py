"""Physical-support-only BNNA adaptation and independent query inference."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tools')]
import numpy as np
from cvsrffi.stage2_d92_bnna import FROZEN_CONFIG,fit_bnna
from export_d92_bnna_features import CACHE_SCHEMA,CACHE_NAME,read,write,validate_capsule,validate_origin,peak_process_rss
from predict_d92_support_cv import validate_split

SOURCE_VALIDATION_REASON='Not run by design: user prohibits auxiliary source-sample and source-feature-cache access'
STEP_FIELDS=['split_id','scope','fold','step','train_k','train_physical_count','active_rank',
    'loss_cls','loss_view','loss_gate','loss_cls_weight','loss_view_weight','loss_gate_weight','loss_total',
    'learning_rate','gradient_norm','gate_before_min','gate_before_max','gate_after_min','gate_after_max',
    'clipped_coordinates','step_seconds','elapsed_seconds','loss_evaluation',
    'near_zero_prototype_count','near_zero_physical_mean_count']


def stable_predictions(scores,classes):
    order=np.asarray(sorted(range(len(classes)),key=lambda index:classes[index]))
    return order[np.argmax(scores[:,order],axis=1)]


def load_features(*,bnna_features,capsule,row_root,expected_capsule_id,expected_checkpoint_sha256,config):
    root=Path(bnna_features)
    marker=read(root/'features_complete.json');provenance=read(root/'checkpoint_provenance.json')
    previous,origin=validate_origin(row_root,capsule,expected_checkpoint_sha256)
    if (marker.get('status')!='BNNA_FEATURES_COMPLETE' or marker.get('schema')!=CACHE_SCHEMA
            or marker.get('capsule_id')!=expected_capsule_id or marker.get('checkpoint_sha256')!=expected_checkpoint_sha256
            or marker.get('algorithm')!=config['algorithm'] or marker.get('classes')!=origin['classes']
            or marker.get('model_seed')!=previous['seed'] or marker.get('dtype')!='float32'
            or marker.get('query_used_for_fitting') is not False or marker.get('source_data_access') is not False
            or marker.get('truth_read') is not False or marker.get('encoder_updated') is not False
            or marker.get('native_eval') is not True or marker.get('native_buffers_unchanged') is not True
            or marker.get('view_count_per_observation')!=4 or marker.get('new_ground_statistics_bytes')!=0
            or provenance.get('checkpoint_sha256')!=expected_checkpoint_sha256 or provenance.get('classes')!=origin['classes']
            or provenance.get('model_seed')!=previous['seed'] or provenance.get('checkpoint_inheritance')!=[]
            or provenance.get('target_access_before_freeze') is not False
            or provenance.get('verdict')!='MATCHED_SOURCE_ONLY_SCRATCH'
            or provenance.get('source_role_comparison')!='EXACT_MATCH' or provenance.get('checkpoint_epoch')!=200
            or type(marker.get('model_file_bytes')) is not int or marker['model_file_bytes']<=0
            or provenance.get('model_file_bytes')!=marker['model_file_bytes']):
        raise ValueError('BNNA feature provenance mismatch')
    feature_path=root/CACHE_NAME
    with np.load(feature_path,allow_pickle=False) as data:
        if set(data.files)!={'identity_views','fft','ids','checkpoint_sha256','capsule_id','algorithm_json'}:
            raise ValueError('Unexpected BNNA cache members')
        if (data['checkpoint_sha256'].item()!=expected_checkpoint_sha256 or data['capsule_id'].item()!=expected_capsule_id
                or json.loads(data['algorithm_json'].item())!=config['algorithm']):
            raise ValueError('BNNA cache binding mismatch')
        identity,fft,ids=data['identity_views'],data['fft'],data['ids'].astype(str)
    with np.load(Path(capsule)/'received.npz',allow_pickle=False) as data:physical_ids=data['ids'].astype(str)
    if (ids.ndim!=1 or not len(ids) or len(set(ids))!=len(ids) or not np.array_equal(ids,physical_ids)
            or identity.shape!=(len(ids),4,160) or fft.shape!=(len(ids),96)
            or identity.dtype!=np.float32 or fft.dtype!=np.float32 or not np.isfinite(identity).all() or not np.isfinite(fft).all()
            or marker.get('count')!=len(ids) or marker.get('identity_views_shape')!=list(identity.shape)
            or marker.get('fft_shape')!=list(fft.shape) or marker.get('identity_views_bytes')!=identity.nbytes
            or marker.get('fft_bytes')!=fft.nbytes or marker.get('feature_array_bytes')!=identity.nbytes+fft.nbytes
            or marker.get('feature_file_bytes')!=feature_path.stat().st_size):
        raise ValueError('BNNA cache physical ID/shape/byte mismatch')
    identity.setflags(write=False);fft.setflags(write=False)
    return identity,fft,ids,marker,previous


def scalar_record(record):
    """Compact tables keep measured scalar fields; full arrays remain in trace."""
    return {key:value for key,value in record.items() if value is None or isinstance(value,(str,int,float,bool))}


def final_fit_summary(record):
    result=scalar_record(record)
    result.update({'final_'+key:value for key,value in scalar_record(record.get('final_losses') or {}).items()})
    return result


def predict(*,row_root,capsule,output,bnna_features,expected_capsule_id,expected_checkpoint_sha256,config):
    if set(config)!={'algorithm'} or config['algorithm']!=FROZEN_CONFIG:raise ValueError('Expected frozen BNNA configuration')
    row_root,capsule,out=map(Path,(row_root,capsule,output))
    if out.exists():raise FileExistsError(out)
    manifest=validate_capsule(capsule,expected_capsule_id)
    identity,fft,ids,marker,previous=load_features(bnna_features=bnna_features,capsule=capsule,row_root=row_root,
        expected_capsule_id=expected_capsule_id,expected_checkpoint_sha256=expected_checkpoint_sha256,config=config)
    old_classes=marker['classes'];paths=sorted((capsule/'splits').glob('*.json'))
    if len(paths)!=manifest['split_count']:raise ValueError('Incomplete split matrix')
    splits=[read(path) for path in paths]
    if len({split['split_id'] for split in splits})!=len(splits):raise ValueError('Duplicate split IDs')
    for split in splits:validate_split(split,manifest,ids,old_classes)
    out.mkdir(parents=True,exist_ok=False)
    payload=dict(new_ground_statistics_bytes=0,model_file_bytes=marker['model_file_bytes'],
        model_already_deployed=None,model_incremental_transfer_bytes=None,
        model_deployment_unknown_reason='Model deployment state is unknown; incremental transfer bytes cannot be inferred',
        model_file_bytes_scope='Complete training checkpoint package on disk, not a minimal inference-only model package',
        received_feature_array_bytes=identity.nbytes+fft.nbytes,received_feature_file_bytes=marker['feature_file_bytes'],
        model_delivery_scope='Existing frozen checkpoint; file bytes reported separately, no new ground statistics',
        received_cache_scope='Locally computed received-only cache, not a ground transmission payload')
    startup=dict(argv=sys.argv,config=config,checkpoint_sha256=expected_checkpoint_sha256,capsule_id=expected_capsule_id,
        query_fit_access=False,truth_read=False,source_data_access=False,ground_summary_access=False,
        source_validation=None,source_validation_reason=SOURCE_VALIDATION_REASON,model_seed=previous['seed'],
        bnna_features=str(bnna_features),payload_audit=payload,prediction_tie_policy='physical_class_id_ascending',
        training_log_unit='actual optimizer step; this algorithm has no separate epoch loop',
        cross_row_adapted_state_reuse=False,new_ground_statistics_bytes=0)
    write(out/'startup.json',startup);print(json.dumps(dict(event='STARTUP',**startup)),flush=True)
    fields=['split_id','completed','total','receiver','scenario','support_seed','new_count','k','classes',
        'selected','selection','candidate_count','fold_count','selected_objective','selected_macro_nll','selected_old_nll','selected_new_nll',
        'fit_seconds','active_rank','head_bytes','basis_bytes','gate_bytes','persistent_state_bytes','final_optimizer_steps',
        'optimizer_steps_recorded','final_step','final_fit','fit_call_seconds','query_score_seconds','prediction_write_seconds',
        'total_seconds','peak_process_rss_bytes','query_rows_used_for_fit','source_rows_used_for_fit',
        'source_validation','source_validation_reason','epoch','epoch_reason']
    with (out/'predictions.jsonl').open('x',encoding='utf-8') as predictions, \
            (out/'fit_trace.jsonl').open('x',encoding='utf-8') as trace, \
            (out/'compact.jsonl').open('x',encoding='utf-8') as compact, \
            (out/'compact.csv').open('x',encoding='utf-8',newline='') as csvfile, \
            (out/'training_steps.jsonl').open('x',encoding='utf-8') as steps_json, \
            (out/'training_steps.csv').open('x',encoding='utf-8',newline='') as steps_csv:
        writer=csv.DictWriter(csvfile,fieldnames=fields);writer.writeheader()
        step_writer=csv.DictWriter(steps_csv,fieldnames=STEP_FIELDS);step_writer.writeheader()
        for index,split in enumerate(splits):
            support,query,labels=validate_split(split,manifest,ids,old_classes);started=time.perf_counter()
            state=fit_bnna(support_identity_views=identity[support],support_fft=fft[support],support_labels=labels,
                support_ids=ids[support],classes=split['registered_classes'],old_classes=old_classes)
            fit_call_seconds=time.perf_counter()-started
            if list(state.classes)!=split['registered_classes']:raise ValueError('BNNA class order mismatch')
            audit=state.audit_dict();query_identity,query_fft=identity[query],fft[query]
            score_started=time.perf_counter();scores=state.score(query_identity,query_fft)
            query_score_seconds=time.perf_counter()-score_started
            if scores.shape!=(len(query),len(state.classes)) or not np.isfinite(scores).all():raise ValueError('Invalid query scores')
            write_started=time.perf_counter()
            record={key:split[key] for key in ('split_id','capsule_id','receiver','scenario','k','support_seed')}
            record.update(mode='d92_bnna_registration',classes=list(state.classes),query_ids=ids[query].tolist(),
                predicted_indices=stable_predictions(scores,state.classes).tolist(),scores=scores.tolist())
            predictions.write(json.dumps(record,allow_nan=False)+'\n');predictions.flush()
            prediction_write_seconds=time.perf_counter()-write_started
            coordinates={key:split[key] for key in ('receiver','scenario','support_seed')}
            coordinates['new_count']=len(state.classes)-len(old_classes)
            audit.update(registered_classes=list(state.classes),old_classes=list(old_classes),**coordinates,
                fit_call_seconds=fit_call_seconds,query_score_seconds=query_score_seconds,prediction_write_seconds=prediction_write_seconds)
            support_classes={str(identifier):split['registered_classes'][int(label)] for identifier,label in zip(ids[support],labels)}
            for assignment in audit['physical_fold_assignment']:
                assignment['class_id']=support_classes[assignment['physical_id']]
            trace.write(json.dumps(dict(split_id=split['split_id'],**audit),allow_nan=False)+'\n');trace.flush()
            for step in audit['steps']:
                small_step=dict(split_id=split['split_id'],**scalar_record(step))
                steps_json.write(json.dumps(small_step,allow_nan=False)+'\n')
                step_writer.writerow({key:'N/A' if value is None else value for key,value in small_step.items()})
                print(json.dumps(dict(event='TRAINING_STEP',**small_step),allow_nan=False),flush=True)
            steps_json.flush();steps_csv.flush()
            final_steps=[step for step in audit['steps'] if step.get('scope')=='final']
            small=dict(split_id=split['split_id'],completed=index+1,total=len(splits),k=split['k'],classes=len(state.classes),**coordinates,
                **{key:audit[key] for key in ('selected','selection','candidate_count','fold_count',
                    'selected_objective','selected_macro_nll','selected_old_nll','selected_new_nll','fit_seconds','active_rank',
                    'head_bytes','basis_bytes','gate_bytes','persistent_state_bytes','final_optimizer_steps')},
                optimizer_steps_recorded=len(audit['steps']),final_step=scalar_record(final_steps[-1]) if final_steps else None,
                final_fit=final_fit_summary(audit['final_fit']),
                fit_call_seconds=fit_call_seconds,query_score_seconds=query_score_seconds,prediction_write_seconds=prediction_write_seconds,
                total_seconds=time.perf_counter()-started,peak_process_rss_bytes=peak_process_rss(),
                query_rows_used_for_fit=0,source_rows_used_for_fit=0,source_validation=None,
                source_validation_reason=SOURCE_VALIDATION_REASON,epoch=None,
                epoch_reason='Actual fixed optimizer steps are logged; no separate epoch loop is defined')
            compact.write(json.dumps(small,allow_nan=False)+'\n');compact.flush()
            csvrow={key:'N/A' if value is None else json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value for key,value in small.items()}
            writer.writerow(csvrow);csvfile.flush();print(json.dumps(dict(event='ROW_COMPLETE',**small),allow_nan=False),flush=True)
    write(out/'predictions_complete.json',dict(status='PREDICTIONS_COMPLETE',split_count=len(splits),predictions=len(splits),
        capsule_id=expected_capsule_id,truth_read=False,source_data_access=False,payload_audit=payload,
        peak_process_rss_bytes=peak_process_rss(),new_ground_statistics_bytes=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('row-root','capsule','output','bnna-features','config','expected-capsule-id','expected-checkpoint-sha256'):
        parser.add_argument('--'+name,required=True)
    args=vars(parser.parse_args());args['config']=read(args['config']);predict(**args)


if __name__=='__main__':main()
