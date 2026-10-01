"""CVS-only source R&D; wait for the existing owner to drain its launch queue."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.cvs_identity_ce.source import write
from experiments.cvs_clean_design.dispatch import choose_gpu
from experiments.cvs_clean_design.model import BASELINES
from experiments.cvs_residual_identity.model import VARIANTS
from experiments.cvs_residual_identity.source import validate_config
SEEDS={2026092701,2026092702,2026092703,2026092704}


def validate_spec(spec):
    if any(k.startswith('p1_') or k.startswith('target') for k in spec):
        raise ValueError('Source R&D dispatcher rejects target paths')
    rows=spec['rows']; expected={(v,s) for v in VARIANTS for s in SEEDS}
    if len(rows)!=8 or {(r['variant'],r['model_seed']) for r in rows}!=expected:
        raise ValueError('Unexpected registered source matrix')
    if len({r['row_id'] for r in rows})!=8 or len({r['source_output'] for r in rows})!=8:
        raise ValueError('Duplicate row/output')
    run=Path(spec['runtime_root']).resolve()
    for row in rows:
        output=Path(row['source_output']).resolve()
        if output!=run/row['row_id']/'source':raise ValueError('Output outside exclusive row')
        c=validate_config(json.loads(Path(row['source_config']).read_text(encoding='utf-8')))
        if c['variant']!=row['variant'] or c['model_seed']!=row['model_seed'] or Path(c['output_root']).resolve()!=output:
            raise ValueError('Registered config mismatch')
    return spec


def predecessor_queue_drained(path):
    # Scheduling prevents two owners racing to launch into the same free slot.
    # Other already-running source jobs remain visible to choose_gpu/occupancy.
    try:state=json.loads(Path(path).read_text(encoding='utf-8'))
    except json.JSONDecodeError:return False  # Existing owner may be mid-write.
    rows=state.get('rows',{})
    if len(rows)!=32 or state.get('target_access') is not False or any(row.get('status') not in {'QUEUED','RUNNING','FAILED','SOURCE_TRAINED'} for row in rows.values()):
        raise ValueError('Unknown predecessor scheduling state')
    return all(row['status']!='QUEUED' for row in rows.values())


def select_source_candidate(records):
    groups={v:[r for r in records if r['variant']==v] for v in VARIANTS}
    if len(records)!=8 or any(len(rows)!=4 or {r['seed'] for r in rows}!=SEEDS for rows in groups.values()):
        raise ValueError('Incomplete source matrix; cannot select')
    summary={v:dict(score=statistics.mean(.5*r['accuracy']+.5*r['worst_rx'] for r in rows),
        source_accuracy=statistics.mean(r['accuracy'] for r in rows),worst_rx_accuracy=statistics.mean(r['worst_rx'] for r in rows),
        parameters=max(r['parameters'] for r in rows),conv_linear_macs=statistics.mean(r['macs'] for r in rows)) for v,rows in groups.items()}
    return frozen_choice(summary)


def frozen_choice(summary):
    eligible=[v for v in summary if summary[v]['parameters']<=1.1*382146]
    if not eligible:raise ValueError('No eligible lightweight CVS candidate')
    best=max(summary[v]['score'] for v in eligible)
    chosen=min((v for v in eligible if summary[v]['score']>=best-.002),key=lambda v:(summary[v]['conv_linear_macs'],summary[v]['parameters'],v))
    return dict(status='SOURCE_SELECTION_FROZEN',selected_variant=chosen,source_summaries=summary,
        selection_rule='four-seed E200 0.5 source-V accuracy+0.5 worst source RX;within0.2pp lower MACs then parameters',
        target_access=False,target_score_used=False,test_view='clean')


def combine_research_selection(predecessor,current):
    for selection in [predecessor,current]:
        if selection['status']!='SOURCE_SELECTION_FROZEN' or selection['target_access'] or selection['target_score_used']:
            raise ValueError('Source selection not valid/frozen')
    expected={'orthogonal_pa','moment_pool','orthogonal_moment','shared_complex',*VARIANTS}
    summary={name:metrics for selection in [predecessor,current] for name,metrics in selection['source_summaries'].items() if name not in BASELINES}
    if set(summary)!=expected:raise ValueError('Incomplete six-candidate research universe')
    result=frozen_choice(summary)
    result.update(test_variants=[*BASELINES,result['selected_variant']],candidate_universe=sorted(expected),
        selection_scope='all six registered source-only CVS candidates before any current target prediction')
    return result


def dispatch(spec_path):
    spec=validate_spec(json.loads(Path(spec_path).read_text(encoding='utf-8')))
    run,logs=Path(spec['runtime_root']),Path(spec['log_root'])
    run.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    state=dict(status='WAITING_PREDECESSOR_QUEUE_DRAIN',pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,run_id=spec['run_id'],
        launch_owner=spec['launch_owner'],commit=(ROOT/'release_commit.txt').read_text().strip(),target_access=False,
        rows={r['row_id']:dict(status='QUEUED',variant=r['variant'],model_seed=r['model_seed']) for r in spec['rows']})
    write(run/'pipeline_state.json',state)
    while not predecessor_queue_drained(spec['predecessor_pipeline']):time.sleep(10)
    state['status']='SOURCE_TRAINING';write(run/'pipeline_state.json',state)
    pending=list(spec['rows']);active={}
    while pending or active:
        for rid,job in list(active.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();del active[rid];entry=state['rows'][rid]
            entry.update(exit_code=code,finished=time.time(),status='FAILED' if code else 'SOURCE_TRAINED')
            if not code:
                p=Path(entry['source_output'])/'completion.json'
                if not p.exists() or json.loads(p.read_text())['epoch']!=200:entry['status']='FAILED'
            write(run/'pipeline_state.json',state)
        if pending:
            gpu=choose_gpu(active)
            if gpu is not None:
                row=pending.pop(0);rid=row['row_id'];logpath=logs/(rid+'.log');log=logpath.open('x')
                command=[sys.executable,'-u','-m','experiments.cvs_residual_identity.source','--config',row['source_config']]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                active[rid]=dict(process=child,gpu=gpu,log=log)
                state['rows'][rid].update(status='RUNNING',pid=child.pid,gpu=gpu,argv=command,log=str(logpath),source_output=row['source_output'],started=time.time())
                write(run/'pipeline_state.json',state);print('LAUNCHED '+json.dumps(state['rows'][rid]),flush=True)
        if pending or active:time.sleep(5)
    if any(r['status']!='SOURCE_TRAINED' for r in state['rows'].values()):
        state['status']='FAILED_PARTIAL';write(run/'pipeline_state.json',state);return 1
    records=[]
    for row in spec['rows']:
        folder=Path(row['source_output']);done=json.loads((folder/'completion.json').read_text());profile=json.loads((folder/'resource_profile.json').read_text())
        m=done['final_source_metrics']
        records.append(dict(variant=row['variant'],seed=row['model_seed'],accuracy=m['source_val_accuracy'],worst_rx=m['source_val_worst_rx'],
            parameters=profile['total_parameters'],macs=profile['conv_linear_macs_per_sample']))
    selection=select_source_candidate(records);write(run/'source_selection.json',selection)
    predecessor=Path(spec['predecessor_selection'])
    if predecessor.exists():
        research=combine_research_selection(json.loads(predecessor.read_text()),selection)
        research['source_selection_refs']=[str(predecessor),str(run/'source_selection.json')]
        write(run/'research_selection.json',research)
        state['status']='SOURCE_RESEARCH_COMPLETE_AWAITING_FROZEN_CLEAN_TEST'
    else:state['status']='SOURCE_RESEARCH_COMPLETE_AWAITING_PREDECESSOR_SELECTION'
    write(run/'pipeline_state.json',state);return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();raise SystemExit(dispatch(a.spec))
