"""Posthoc source-only error complementarity of fixed NoSIG raw classifiers."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .restoration_labeler_policy import probability,quality_pass

RUN='20261026-analysis-PAMF-NoSIG-RawAnchor-labels-r01'
PAIRS=(('SupAnchor','sup_logits'),('PLAnchor','pl_logits'))


def certify_pair(adapted_logits,anchor_logits,original_confidence):
    r=np.asarray(adapted_logits);a=np.asarray(anchor_logits);old=np.asarray(original_confidence)
    if (r.ndim!=2 or len(r)<1 or r.shape[1]!=6 or a.shape!=r.shape or old.shape!=(len(r),)
            or not np.isfinite(old).all() or ((old<0)|(old>1)).any()):
        raise ValueError('Aligned six-class frozen raw logits and original source hard pool required')
    rp=probability(r);ap=probability(a);label=rp.argmax(1)
    return dict(accepted=(old<.95)&(rp.max(1)>=.95)&(ap.max(1)>=.95)&(label==ap.argmax(1)),label=label,
        adapted_probability=rp,anchor_probability=ap)


def count(name,mask,pred,y,old):
    n=int(mask.sum());correct=int((mask&(pred==y)).sum());fixed=int((mask&(old!=y)&(pred==y)).sum());broken=int((mask&(old==y)&(pred!=y)).sum())
    return dict(method=name,certified=n,correct=correct,precision=100*correct/n if n else None,corrected=fixed,broken=broken,net_correct=fixed-broken)


def write_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def run(inputs,output,commit):
    source=Path(inputs);out=Path(output);manifest=json.loads((source/'input_manifest.json').read_text(encoding='utf-8'))
    if (manifest['run_id']!=RUN or manifest['source_role']!='V' or manifest['records']!=27000 or manifest['model_seed']!=2026092701
            or manifest['U_truth_read'] or manifest['target_reads'] or manifest['new_model_forwards'] or manifest['IQ_reads']):
        raise ValueError('Existing source V predictions only; no U/target/IQ/model execution')
    out.mkdir(parents=True,exist_ok=False)
    with np.load(source/'frozen_source_predictions.npz',allow_pickle=False) as a:
        if set(a.files)!={'adapted_logits','sup_logits','pl_logits','original_confidence','original_label','source_y','source_rx','source_day','ids'}:
            raise ValueError('One source-only frozen posterior bundle required')
        raw=a['adapted_logits'];old_conf=a['original_confidence'];ids=a['ids']
        if raw.shape!=(27000,6) or old_conf.shape!=(27000,) or len(ids)!=27000:raise ValueError('Original complete source V order required')
        decisions={name:certify_pair(raw,a[key],old_conf) for name,key in PAIRS}
        np.savez(out/'frozen_V_anchor_predictions.npz',methods=np.asarray([name for name,_ in PAIRS]),ids=ids,
            accepted=np.stack([decisions[name]['accepted'] for name,_ in PAIRS]),label=np.stack([decisions[name]['label'] for name,_ in PAIRS]))
        for name,_ in PAIRS:
            (out/name).mkdir(exist_ok=False)
            np.savez(out/name/'frozen_V_predictions.npz',accepted=decisions[name]['accepted'],label=decisions[name]['label'],ids=ids)
        # All candidate decisions are fixed before the visible source truth-side arithmetic.
        y=a['source_y'];old=a['original_label'];rx=a['source_rx'];day=a['source_day']
        if (y.shape!=(27000,) or old.shape!=(27000,) or ((y<0)|(y>=6)).any() or ((old<0)|(old>=6)).any()
                or set(rx.tolist())-{1,3,4,6,8} or set(day.tolist())-{1,2,3}):raise ValueError('Visible original F0 source V labels and domains required')
        hard=old_conf<.95;rows=[];full=[];cells=[];overlaps=[]
        probabilities={'RawLabeler':probability(raw),'NoSIG-Sup':probability(a['sup_logits']),'NoSIG-PL':probability(a['pl_logits'])}
        for name,p in probabilities.items():
            pred=p.argmax(1);rows.append(count(name,hard&(p.max(1)>=.95),pred,y,old))
            full.append(dict(method=name,source_V_accuracy=100*float((pred==y).mean()),hard_pool_accuracy=100*float((pred[hard]==y[hard]).mean())))
        rpred=probabilities['RawLabeler'].argmax(1)
        for name,key in PAIRS:
            d=decisions[name];row=count(name,d['accepted'],d['label'],y,old)
            row['quality_pass']=quality_pass(row['certified'],row['correct'],row['net_correct']);rows.append(row)
            write_json(out/name/'source_quality.json',row)
            anchor=probability(a[key]).argmax(1)
            overlaps.append(dict(method=name,hard_pool=int(hard.sum()),both_correct=int((hard&(rpred==y)&(anchor==y)).sum()),
                adapted_only_correct=int((hard&(rpred==y)&(anchor!=y)).sum()),anchor_only_correct=int((hard&(rpred!=y)&(anchor==y)).sum()),
                both_wrong_same_class=int((hard&(rpred!=y)&(anchor!=y)&(rpred==anchor)).sum()),both_wrong_different_class=int((hard&(rpred!=y)&(anchor!=y)&(rpred!=anchor)).sum())))
            for dim,value in [('overall',np.zeros(len(y),int)),('RX',rx),('day',day),('TX',y)]:
                for item in np.unique(value):
                    cell=count(name,d['accepted']&(value==item),d['label'],y,old)
                    cell.update(dimension=dim,stratum='ALL' if dim=='overall' else str(item));cells.append(cell)
    for name,table in [('source_label_quality.csv',rows),('source_quality_cells.csv',cells),('source_accuracies.csv',full),('source_error_overlap.csv',overlaps)]:
        fields=list(dict.fromkeys(k for row in table for k in row))
        with (out/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(table)
    result=dict(status='SOURCE_POSTERIOR_ANALYSIS_COMPLETE',run_id=RUN,code_commit=commit,model_seed=2026092701,source_V=27000,hard_pool=int(hard.sum()),
        rule=dict(confidence=.95,minimum_certified=200,minimum_precision_percent=99.5,minimum_net_correct=100),
        rows=rows,source_accuracies=full,error_overlaps=overlaps,source_cells=cells,source_manifest=manifest,
        U_truth_read=False,target_reads=0,IQ_reads=0,new_model_forwards=0,new_training=False,automatic_student_launch=False,
        scope='Fixed source posterior complementarity only; no target or U precision claim')
    write_json(out/'complete.json',result)
    print(json.dumps({k:result[k] for k in ('status','rows','source_accuracies','error_overlaps')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();run(a.inputs,a.output,a.commit)
