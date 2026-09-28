"""Independent arithmetic audit of every complete confusion matrix; no fitting."""
import argparse
import json
import math
from pathlib import Path


def audit(data, expected_count):
    if data.get('status')!='SCORED' or len(data['results'])!=expected_count:
        raise ValueError('Incomplete scored matrix')
    for row in data['results']:
        cm=row['confusion'];c=len(row['classes'])
        if (len(cm)!=c or c<6 or row['class_count']!=c or row['new_count']!=c-6
                or any(len(line)!=c or any(type(v) is not int or v<0 for v in line) for line in cm)):
            raise ValueError('Invalid confusion matrix')
        counts=[sum(line) for line in cm]
        if min(counts)<1:raise ValueError('Missing class query observations')
        total=sum(counts);correct=sum(cm[i][i] for i in range(c))
        old_total=sum(counts[:6]);old_correct=sum(cm[i][i] for i in range(6))
        old=old_correct/old_total;new=(correct-old_correct)/(total-old_total) if c>6 else None
        h=2*old*new/(old+new) if new is not None and old+new else (0.0 if new is not None else None)
        f1=[2*cm[i][i]/(counts[i]+sum(line[i] for line in cm)) for i in range(c)]
        ca=[cm[i][i]/counts[i] for i in range(c)]
        expected=dict(accuracy=correct/total,old_accuracy=old,new_accuracy=new,harmonic_mean=h,
            macro_f1=sum(f1)/c,old_macro_f1=sum(f1[:6])/6,new_macro_f1=sum(f1[6:])/(c-6) if c>6 else None,
            old_floor=min(ca[:6]),new_floor=min(ca[6:]) if c>6 else None)
        for key,value in expected.items():
            actual=row[key]
            if ((value is None and actual is not None) or (value is not None and
                    (not isinstance(actual,(int,float)) or not math.isfinite(actual) or not math.isclose(actual,value,rel_tol=0,abs_tol=1e-12)))):
                raise ValueError(f'Metric arithmetic mismatch: {row["split_id"]}/{key}')
        if row['query_count']!=total or len(row['class_accuracy'])!=c or any(not math.isclose(a,b,rel_tol=0,abs_tol=1e-12) for a,b in zip(row['class_accuracy'],ca)):
            raise ValueError('Query count or per-class arithmetic mismatch')
    return dict(status='VERIFIED',records=expected_count,scope='All confusion matrices independently recomputed: accuracy,old/new,H,macroF1/groupF1,floors,class_accuracy,query totals',
        candidate_selection=False,goal_complete=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','scores','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    spec=json.loads(args.spec.read_text(encoding='utf-8'));data=json.loads(args.scores.read_text(encoding='utf-8'))
    result=audit(data,spec['confirmation']['predictions_total'])
    result.update(run_id=spec['run_id'],raw_scores=str(args.scores),raw_scores_bytes=args.scores.stat().st_size)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
