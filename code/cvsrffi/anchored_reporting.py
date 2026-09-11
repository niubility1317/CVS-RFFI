"""Inspectable source-development counts and synchronized stage timings."""
import io,json,platform,time
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
    for name,key in (('A6','nested_log_probabilities'),('A5_inner_selected','nested_fixed_log_probabilities')):
        prediction=nested[key].argmax(-1)
        for group,value,index in groups:
            record=dict(candidate=name,group=group,value=value,**count_changes(rows.labels[index],baseline[index],prediction[index],paired_weights(rows)[index]))
            records.append(record)
        flows.extend(dict(candidate=name,**r) for r in class_flow(rows.labels,prediction,rows.baseline_inference_logits.shape[1]))
    write_csv(Path(path)/'nested_source_metrics.csv',records);write_csv(Path(path)/'nested_class_flow.csv',flows)
    return records


def assess_source_promotion(seed_reports,required_seeds=(392005,392006,392007)):
    """Conservative source-development criterion; never confirmation evidence."""
    if set(seed_reports)!=set(required_seeds):return dict(status='PENDING_ALL_HEAD_SEEDS',passed=False,available_seeds=sorted(seed_reports))
    evidence=[]
    for seed,records in sorted(seed_reports.items()):
        rows=[r for r in records if r['candidate']=='A6'];all_rows=[r for r in rows if r['group']=='all']
        rx=[r for r in rows if r['group']=='RX'];tx=[r for r in rows if r['group']=='TX'];clean=[r for r in rows if r['group']=='view' and r['value']=='clean']
        complete=len(all_rows)==1 and len(rx)==5 and len(tx)>0 and len(clean)==1
        passed=complete and all_rows[0]['weighted_net']>0 and sum(r['weighted_net']>=0 for r in rx)>=4 and clean[0]['weighted_net']>=-.002 and min(r['weighted_net'] for r in rx+tx)>=-.005
        evidence.append(dict(head_seed=seed,passed=bool(passed),group_counts=dict(RX=len(rx),TX=len(tx))))
    return dict(status='SOURCE_DEVELOPMENT_SUPPORTED' if all(r['passed'] for r in evidence) else 'SOURCE_CRITERION_NOT_MET',
                passed=all(r['passed'] for r in evidence),seeds=evidence,confirmation=False,backbone_seed_replications=1)


def fusion_reports(rows,actions,path,*,candidate='A5',lambda_h=2.):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    baseline=rows.baseline_inference_logits.argmax(-1);records=[];flows=[]
    groups=[('all','all',torch.arange(len(rows)))]
    for key,tensor in (('RX',rows.receiver),('TX',rows.labels),('day',rows.day)):
        groups.extend((key,str(int(v)),torch.where(tensor==v)[0]) for v in tensor.unique(sorted=True))
    for view in sorted(set(rows.view_ids)):
        groups.append(('view',view,torch.tensor([i for i,v in enumerate(rows.view_ids) if v==view])))
    for action,alpha in enumerate(ALPHAS):
        prediction=actions['log_probabilities'][:,action].argmax(-1).cpu()
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
