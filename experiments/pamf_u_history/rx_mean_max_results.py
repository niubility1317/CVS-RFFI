"""Full three-seed maximum-style comparison versus three reused reference methods."""
import argparse
import itertools
import json
from pathlib import Path,PurePosixPath
from .rx_mean_max_source import RUN,PREDECESSOR,BASELINE,METHODS,SEEDS
from .covariance_three_results import summarize as summarize_matrix
from .rx_mean_results import collection_script as matrix_script
from .fonly_results import save

REFERENCES=('SIG1-PL','SIG1-Sup','RXMean-U')


def expected_run(method,seed):
    if method not in (*METHODS,*REFERENCES) or seed not in SEEDS:raise ValueError('Exact new max-style and three declared reused references required')
    if method in ('SIG1-PL','SIG1-Sup'):return BASELINE
    return PREDECESSOR if method=='RXMean-U' else RUN


def locations(rt):
    base=PurePosixPath(rt['project'])/'runs';rows=[]
    for method,seed in itertools.product((*METHODS,*REFERENCES),SEEDS):
        rid='F0-'+method+'-s'+str(seed);run=expected_run(method,seed)
        rows.append(dict(method=method,model_seed=seed,run_id=run,row_id=rid,root=str(base/run/'readouts'/rid)))
    return rows


def collection_script(rt):
    return matrix_script(rt,run_id=RUN,methods=METHODS,references=REFERENCES,row_locations=locations(rt),
        completion_status='RX_MEAN_MAX_THREE_SEED_COMPLETE',models=3,jobs=12)


def summarize(bundle):
    return summarize_matrix(bundle,methods=METHODS,references=REFERENCES,run_id=RUN,row_run=expected_run,status='ANALYZED_RX_MEAN_MAX_THREE_SEED_FONLY')


def collect(runtime,output):
    from experiments.cric_dura_comparison.publish import ssh
    out=Path(output)
    if out.exists():raise FileExistsError('Preserve previous frozen decision/score readback')
    rt=json.loads(Path(runtime).read_text(encoding='utf-8'));r=ssh(collection_script(rt));summarize(r)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=r['status'],score_rows=len(r['scores']),physical_pairings=len(r['pairing_checks']),candidate_raw_decisions=r['logical_candidate_raw_decisions'])))


def analyze(bundle,output):
    result,tables=summarize(json.loads(Path(bundle).read_text(encoding='utf-8')));out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,rows in tables.items():save(out/(name+'.csv'),rows)
    print(json.dumps(dict(status=result['status'],overall=[r for r in result['summary'] if r['population']=='overall_168000'],full_goal_complete=False)))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('collect');c.add_argument('--runtime',required=True);c.add_argument('--output',required=True)
    a=sub.add_parser('analyze');a.add_argument('--bundle',required=True);a.add_argument('--output',required=True)
    args=p.parse_args();collect(args.runtime,args.output) if args.action=='collect' else analyze(args.bundle,args.output)
