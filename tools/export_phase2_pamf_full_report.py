"""Read the complete already-scored PAMF/SCALR MIX matrix, without query data."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.run);complete=read(root/'complete.json');fixed=read(root/'predictions_complete.json');scorestate=read(root/'score_state.json')
    assert complete['status']=='SCORED_COMPLETE' and complete['truth_last'] is True and complete['target_feedback'] is False
    assert fixed['status']=='ALL_PREDICTIONS_FIXED' and fixed['truth_read'] is False and fixed['query_fit'] is False
    assert scorestate['status']=='COMPLETE' and not scorestate['failed'] and not scorestate['active']
    ks=(1,5,10,20);ns=(2,5,10,20)
    expected={('A0',path,k,0) for path in ('Q0','Q1') for k in ks}
    expected|={('B',path,k,0) for path in ('S0Q0','S1Q1') for k in ks}
    expected|={('C',path,k,n) for path in ('S0Q0','S1Q1') for k in ks for n in ns}
    rows=[];coverage=Counter();tasks=0;identities=set()
    for task in sorted((root/'tasks').glob('s*-c*')):
        seed=int(task.name[1:11])
        if seed not in (2026092701,2026092702,2026092703):continue
        paths=list((task/'scores').glob('*/metrics.json'));assert len(paths)==1
        data=read(paths[0]);done=read(paths[0].parent/'complete.json')
        assert done['status']=='SCORED_COMPLETE' and done['truth_last'] is True and done['target_feedback'] is False
        assert data['truth_last'] is True and data['target_feedback'] is False and len(data['rows'])==done['rows']
        seen=set();taskident=set()
        for row in data['rows']:
            if row.get('episode') not in tuple(f'MIX{i:02}' for i in range(1,9)):continue
            if not ((row['stage']=='A0' and row['arm']=='LocalRidge' and row['input_path'] in ('Q0','Q1')) or
                    (row['stage'] in ('B','C') and row['arm']=='R42' and row['input_path'] in ('S0Q0','S1Q1'))):continue
            cell=(row['stage'],row['input_path'],row['k'],row['new_count'])
            assert cell in expected and cell not in seen;seen.add(cell)
            assert row['model_seed']==seed
            cap=f"capsules/{row['physical_pool']}/{row['episode']}/{row['receiver']}"
            taskident.add((cap,seed));m=row['metrics']
            rows.append(dict(group='pamf',profile='PAMF',method='PAMF_clean' if row['input_path'] in ('Q0','S0Q0') else 'PAMF_恢复',
                capsule=cap,seed=seed,stage=row['stage'],k=row['k'],n=row['new_count'],split_id=row['split_id'],
                registered_classes=row['registered_classes'],query_count=m['count'],arm=row['arm'],accuracy=m['accuracy'],
                macro_accuracy=m['macro_accuracy'],old=m.get('old_macro_accuracy'),new=m.get('new_macro_accuracy'),
                h=m.get('H_macro'),f1=m.get('macro_f1')))
            coverage[cell]+=1
        assert seen==expected and len(taskident)==1
        identity=next(iter(taskident));assert identity not in identities;identities.add(identity);tasks+=1
    assert tasks==504 and len(rows)==24192 and all(coverage[cell]==504 for cell in expected)
    out=dict(tasks=tasks,errors=[],paired_rows=rows,source_run=str(root),scored_truth_last=True,
             coverage={str(k):v for k,v in coverage.items()})
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False),encoding='utf8')
    print(json.dumps(dict(tasks=tasks,rows=len(rows),cells=len(coverage),each_cell=504,status='VERIFIED'),ensure_ascii=False))


if __name__=='__main__':main()
