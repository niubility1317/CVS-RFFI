"""Inspectable source-development counts and synchronized stage timings."""
import io,json,platform,time,math
from pathlib import Path
import torch
from .anchored_fit import write_csv,paired_weights
from .anchored_fusion import ALPHAS


def count_changes(labels,baseline,prediction,weights=None,lambda_h=2.):
    y=labels.cpu();base=baseline.cpu();pred=prediction.cpu();n=len(y)
    if n==0 or base.shape!=y.shape or pred.shape!=y.shape:raise ValueError('nonempty matching prediction rows required')
    w=torch.ones(n,dtype=torch.double)/n if weights is None else weights.cpu().double()/weights.sum()
    correct=pred==y;before=base==y;rescue=correct&~before;harm=~correct&before;changed=pred!=base
    r=int(rescue.sum());h=int(harm.sum())
    return dict(rows=n,baseline_correct=int(before.sum()),correct=int(correct.sum()),rescue=r,harm=h,net_correct=r-h,
                rescue_fraction=r/n,harm_fraction=h/n,rescue_of_baseline_errors=r/int((~before).sum()) if (~before).any() else None,
                harm_of_baseline_correct=h/int(before.sum()) if before.any() else None,utility=r-lambda_h*h,
                changed=int(changed.sum()),changed_fraction=float(changed.double().mean()),
                weighted_accuracy=float(w[correct].sum()),weighted_baseline_accuracy=float(w[before].sum()),
                weighted_net=float(w[rescue].sum()-w[harm].sum()),weighted_utility=float(w[rescue].sum()-lambda_h*w[harm].sum()))


def class_flow(labels,prediction,class_count):
    labels=labels.cpu();prediction=prediction.cpu();result=[]
    for y in range(class_count):
        truth=labels==y;chosen=prediction==y;tp=int((truth&chosen).sum());total=int(truth.sum());count=int(chosen.sum())
        result.append(dict(class_index=y,prediction_share=count/len(labels),precision=tp/count if count else None,
                           recall=tp/total if total else None,inflow_errors=int((chosen&~truth).sum()),
                           support=total,predicted=count,confusion=json.dumps([int((truth&(prediction==k)).sum()) for k in range(class_count)])))
    return result


def risk_coverage(confidence,correct,weights=None):
    confidence=confidence.cpu().double();correct=correct.cpu();n=len(confidence)
    w=torch.ones(n,dtype=torch.double)/n if weights is None else weights.cpu().double()/weights.sum()
    order=confidence.argsort(descending=True,stable=True);values=confidence[order]
    end=torch.cat((values[1:]!=values[:-1],torch.ones(1,dtype=torch.bool)))
    indices=end.nonzero().flatten();coverage=w[order].cumsum(0);errors=(w[order]*~correct[order]).cumsum(0)
    return [dict(threshold=float(values[i]),coverage=float(coverage[i]),risk=float(errors[i]/coverage[i]),accepted_rows=i+1) for i in indices.tolist()]


def nested_reports(rows,nested,path):
    baseline=rows.baseline_inference_logits.argmax(-1);records=[];flows=[]
    groups=[('all','all',torch.arange(len(rows)))]
    for key,tensor in (('RX',rows.receiver),('TX',rows.labels),('day',rows.day)):
        groups.extend((key,str(int(v)),torch.where(tensor==v)[0]) for v in tensor.unique(sorted=True))
    groups.append(('view','clean',torch.tensor([i for i,v in enumerate(rows.view_ids) if v=='clean'])))
    for name,key in (('A6','nested_predictions'),('A5_inner_selected','nested_fixed_predictions')):
        prediction=nested[key]
        for group,value,index in groups:
            record=dict(candidate=name,group=group,value=value,**count_changes(rows.labels[index],baseline[index],prediction[index],paired_weights(rows)[index]))
            records.append(record)
        flows.extend(dict(candidate=name,**r) for r in class_flow(rows.labels,prediction,rows.baseline_inference_logits.shape[1]))
    write_csv(Path(path)/'nested_source_metrics.csv',records);write_csv(Path(path)/'nested_class_flow.csv',flows)
    return records


def assess_source_promotion(seed_reports,required_seeds=(392005,392006,392007),*,
                            required_rx=(1,3,4,6,8),required_tx=range(6),required_days=(1,2,3)):
    """Conservative source-development criterion; never confirmation evidence."""
    if set(seed_reports)!=set(required_seeds):return dict(status='PENDING_ALL_HEAD_SEEDS',passed=False,available_seeds=sorted(seed_reports))
    evidence=[];comparisons=[];errors=[]
    expected={(group,str(value)) for group,values in (('all',('all',)),('RX',required_rx),('TX',required_tx),
              ('day',required_days),('view',('clean',))) for value in values}
    for seed,records in sorted(seed_reports.items()):
        indexed={};seed_errors=[]
        for candidate in ('A6','A5_inner_selected'):
            rows=[r for r in records if r.get('candidate')==candidate]
            keys=[(r.get('group'),str(r.get('value'))) for r in rows]
            if len(keys)!=len(set(keys)) or set(keys)!=expected:
                seed_errors.append(candidate+': missing, duplicate or unexpected groups');continue
            for row in rows:
                required=('weighted_net','weighted_accuracy','weighted_utility','rows')
                if any(not isinstance(row.get(k),(int,float)) or isinstance(row.get(k),bool) or not math.isfinite(row[k]) for k in required):
                    seed_errors.append(candidate+': missing or nonfinite metrics');break
                if row['rows']<=0 or row['rows']!=int(row['rows']) or not 0<=row['weighted_accuracy']<=1 or not -1<=row['weighted_net']<=1:
                    seed_errors.append(candidate+': invalid metric range');break
                baseline=row['weighted_accuracy']-row['weighted_net']
                # v1 lambda_H=2: net=rescue-harm, utility=rescue-2*harm.
                harm=row['weighted_net']-row['weighted_utility'];rescue=2*row['weighted_net']-row['weighted_utility']
                if not -1e-8<=baseline<=1+1e-8 or harm < -1e-8 or harm>baseline+1e-8 or rescue < -1e-8 or rescue>1-baseline+1e-8:
                    seed_errors.append(candidate+': impossible baseline/rescue/harm/utility');break
                if any(isinstance(v,(int,float)) and not math.isfinite(v) for v in row.values()):
                    seed_errors.append(candidate+': nonfinite metric');break
            if seed_errors:continue
            indexed[candidate]=dict(zip(keys,rows))
            total=indexed[candidate]['all','all']['rows']
            if any(sum(r['rows'] for r in rows if r['group']==group)!=total for group in ('RX','TX','day')) or indexed[candidate]['view','clean']['rows']>total:
                seed_errors.append(candidate+': group counts do not partition all rows')
        if seed_errors:
            errors.append(dict(head_seed=seed,reasons=seed_errors));continue
        a6=indexed['A6'];a5=indexed['A5_inner_selected']
        for key in sorted(expected):
            left,right=a6[key],a5[key]
            if left['rows']!=right['rows'] or abs((left['weighted_accuracy']-left['weighted_net'])-(right['weighted_accuracy']-right['weighted_net']))>1e-8:
                seed_errors.append('A6/A5 group population or H0 baseline mismatch');break
            comparisons.append(dict(head_seed=seed,group=key[0],value=key[1],
                accuracy_delta_A6_vs_A5=left['weighted_accuracy']-right['weighted_accuracy'],
                utility_delta_A6_vs_A5=left['weighted_utility']-right['weighted_utility'],
                net_A6_vs_H0=left['weighted_net'],net_A5_vs_H0=right['weighted_net']))
        if seed_errors:
            errors.append(dict(head_seed=seed,reasons=seed_errors));continue
        rx=[r for (group,_),r in a6.items() if group=='RX'];tx=[r for (group,_),r in a6.items() if group=='TX']
        passed=a6['all','all']['weighted_net']>0 and sum(r['weighted_net']>=0 for r in rx)>=4 and a6['view','clean']['weighted_net']>=-.002 and min(r['weighted_net'] for r in rx+tx)>=-.005
        evidence.append(dict(head_seed=seed,passed=bool(passed),group_counts=dict(RX=len(rx),TX=len(tx),day=len([k for k in a6 if k[0]=='day']))))
    if errors:return dict(status='INCOMPLETE_SOURCE_EVIDENCE',passed=False,errors=errors,confirmation=False)
    return dict(status='SOURCE_DEVELOPMENT_SUPPORTED' if all(r['passed'] for r in evidence) else 'SOURCE_CRITERION_NOT_MET',
                passed=all(r['passed'] for r in evidence),seeds=evidence,comparisons=comparisons,confirmation=False,backbone_seed_replications=1)


def fusion_reports(rows,actions,path,*,candidate='A5',lambda_h=2.):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    baseline=rows.baseline_inference_logits.argmax(-1);records=[];flows=[]
    groups=[('all','all',torch.arange(len(rows)))]
    for key,tensor in (('RX',rows.receiver),('TX',rows.labels),('day',rows.day)):
        groups.extend((key,str(int(v)),torch.where(tensor==v)[0]) for v in tensor.unique(sorted=True))
    for view in sorted(set(rows.view_ids)):
        groups.append(('view',view,torch.tensor([i for i,v in enumerate(rows.view_ids) if v==view])))
    for action,alpha in enumerate(ALPHAS):
        prediction=actions['predictions'][:,action].cpu()
        for group,value,index in groups:
            # Subgroups retain original physical weights then renormalize; a LEO-only subgroup is legal.
            weights=paired_weights(rows,True)[index]
            record=dict(candidate=candidate,action=action,raw_alpha=alpha,group=group,value=value,
                        **count_changes(rows.labels[index],baseline[index],prediction[index],weights,lambda_h))
            record.update(realized_alpha_mean=float(actions['alpha'][index,action].mean()),
                          protection_backoffs=int((actions['reason'][index,action]==1).sum()),
                          ties=int((actions['reason'][index,action]==2).sum()),invalid=int((actions['reason'][index,action]==3).sum()))
            records.append(record)
        flows.extend(dict(candidate=candidate,action=action,**r) for r in class_flow(rows.labels,prediction,rows.baseline_inference_logits.shape[1]))
    write_csv(path/'fusion_audit.csv',records);write_csv(path/'class_flow.csv',flows)
    return records


def profile_stages(functions,*,warmup=3,repeats=10,device='cpu',cache_hit=False):
    if warmup<0 or repeats<1 or not functions:raise ValueError('invalid profile budget')
    cuda=torch.device(device).type=='cuda';results={}
    def sync():
        if cuda:torch.cuda.synchronize(device)
    for name,fn in functions.items():
        for _ in range(warmup):fn()
        sync()
        if cuda:torch.cuda.reset_peak_memory_stats(device)
        times=[]
        for _ in range(repeats):
            sync();start=time.perf_counter();fn();sync();times.append(time.perf_counter()-start)
        results[name]=dict(seconds_mean=sum(times)/len(times),seconds_min=min(times),seconds_max=max(times),
                           samples_seconds=times,warmup=warmup,repeats=repeats,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None)
    return dict(stages=results,device=str(device),pytorch=torch.__version__,python=platform.python_version(),platform=platform.platform(),
                cache_hit=bool(cache_hit),cuda_synchronized=cuda,torch_threads=torch.get_num_threads(),
                load_scope='controlled local call; no claim of exclusive GPU ownership',cuda_available=torch.cuda.is_available())


def save_json(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def partial_pair_diagnostics(head):
    from .pairwise_evidence import schur_incremental_information
    covariance=torch.diag(head.diagonal.double())+head.factor.double()@head.factor.double().T
    records=[];d=covariance.shape[0];start=0
    for name,width in zip(head.pattern_spec.block_names,head.pattern_spec.block_sizes):
        added=list(range(start,start+width));existing=[i for i in range(d) if i not in added];start+=width
        for a in range(len(head.means)):
            for b in range(a+1,len(head.means)):
                result=schur_incremental_information((head.means[a]-head.means[b]).double(),covariance,existing,added)
                records.append(dict(added_block=name,class_a=a,class_b=b,**{k:float(v) for k,v in result.items()},
                                    interpretation='shared source covariance diagnostic, not a class-specific gate'))
    return records
