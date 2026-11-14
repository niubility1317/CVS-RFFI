"""Score fixed paired ISSL budgets and report each receiver separately."""
import argparse
from pathlib import Path

from .clean_issl_short_plan import BUDGETS
from .clean_score import score_job, verify_all
from .io import read, require, write


def score(reference_config,short_config,output):
    reference=read(reference_config)
    short=read(short_config)
    reference_jobs=[entry for entry in reference['jobs'] if entry['group']=='native_cil'
                    and read(entry['job']).get('profile')=='issl_fixed_v1']
    require(len(reference_jobs)==5 and len(short['jobs'])==20,'Wrong comparison size')
    reference=dict(reference,jobs=reference_jobs)
    fixed=verify_all(reference,dict(jobs=5,blocked=[]))
    short_fixed=verify_all(short,dict(jobs=20,blocked=[]))
    identity={(v['job']['capsule'],v['job']['seed'],tuple(map(tuple,v['job']['points']))) for v in fixed}
    require(len(identity)==5,'Duplicate long-budget tasks')
    expected={(i,b) for i in identity for b in BUDGETS}
    observed=set()
    for item in short_fixed:
        job=item['job']
        item_identity=(job['capsule'],job['seed'],tuple(map(tuple,job['points'])))
        budget=tuple(job['issl_budget'])
        require(tuple(item['payload']['issl_budget'])==budget
                and tuple(read(item['entry']['completion'])['issl_budget'])==budget,
                'Completed prediction budget mismatch')
        require((item_identity,budget) not in observed,'Duplicate short-budget prediction')
        observed.add((item_identity,budget))
    require(observed==expected,'Paired budget predictions incomplete')
    root=Path(output).resolve()
    require(not root.exists(),'Score output already exists')
    root.mkdir(parents=True,exist_ok=False)
    rows=[]
    for budget,group in [((2000,300),fixed),*[(b,[x for x in short_fixed
                                                     if tuple(x['job']['issl_budget'])==b]) for b in BUDGETS]]:
        for item in group:
            label=f"ssl{budget[0]}-transfer{budget[1]}-{item['entry']['id']}"
            folder=root/'details'/label
            score_job(item,folder)
            for row in read(folder/'metrics.json')['rows']:
                if row['stage']=='C':
                    rows.append(dict(budget=list(budget),capsule=item['job']['capsule'],
                                     receiver=row['receiver'],k=row['k'],new_count=row['new_count'],
                                     seed=row['model_seed'],metrics=row['metrics']))
    require(len(rows)==25,'Incomplete score rows')
    write(root/'paired_metrics.json',dict(status='SCORED_COMPLETE',rows=rows,truth_last=True,
          target_test_previously_exposed=True,selection_performed=False))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-config',required=True)
    parser.add_argument('--short-config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print('SCORED',len(score(args.reference_config,args.short_config,args.output)))


if __name__=='__main__':main()
