"""Reuse verified immutable closed rows; audit only newly closed source logs."""
import argparse
from importlib import import_module
import json
from pathlib import Path
from .archive_evidence import read_expanded

PACKAGES={'continuation':'experiments.cvs_e20_boundary_style','scratch':'experiments.cvs_scratch_boundary_style'}


def collect(kind, previous_path):
    package=PACKAGES[kind];d=import_module(package+'.design');reader=import_module(package+'.analyze_source')
    previous_path=Path(previous_path)
    if not previous_path.exists():
        return reader.collect()
    previous=read_expanded(previous_path)
    if (previous['run_id']!=d.RUN or previous['errors'] or not previous['source_only']
            or previous['target_prediction_or_scores_read'] is not False
            or not previous['full_closed_source_step_epoch_CSV_warmup_and_stdout_read']):
        raise ValueError('Prior source audit is not reusable')
    registered={r['row_id']:r for r in d.rows()};reused={}
    for row in previous['rows']:
        rid=row['row_id']
        if (rid not in registered or rid in reused or row['status']!='COMPLETE' or row['steps']!=4440
                or row['arm']!=registered[rid]['arm'] or row['seed']!=registered[rid]['model_seed']
                or not row['source_only'] or not row['source_stdout']['closed_source_segment']
                or row['source_stdout']['target_segment_read'] or row['stdout_errors']):
            raise ValueError('A prior closed source row has inconsistent identity or scope')
        reused[rid]=row
    if previous['expected_rows']!=len(registered) or previous['total_steps']!=4440*len(reused):
        raise ValueError('Prior matrix counts differ')
    configs=[d.config(r) for r in d.rows() if r['row_id'] not in reused]
    if not configs:
        return previous
    from experiments.cvs_phase1_stack.publish import ssh
    own=Path(reader.__file__).read_text(encoding='utf-8').split('# LOCAL CONTROL:')[0]
    payload=own+'\nprint(json.dumps(analyze('+repr(d.BASE.as_posix())+','+repr(configs)+'),allow_nan=False))\n'
    fresh=json.loads(ssh(payload))
    if (fresh['run_id']!=d.RUN or fresh['expected_rows']!=len(configs) or fresh['errors']
            or not fresh['source_only'] or fresh['target_prediction_or_scores_read']
            or not fresh['full_closed_source_step_epoch_CSV_warmup_and_stdout_read']):
        raise ValueError('New source evidence invalid: '+repr(fresh['errors']))
    new={r['row_id']:r for r in fresh['rows']}
    if (len(new)!=len(fresh['rows']) or set(new)&set(reused)
            or set(new)|{r['row_id'] for r in fresh['pending']}!={c['row_id'] for c in configs}):
        raise ValueError('Fresh source identities differ from the unread rows')
    combined={**reused,**new}
    rows=[combined[r['row_id']] for r in d.rows() if r['row_id'] in combined]
    result=dict(fresh,status='PARTIAL' if fresh['pending'] else 'COMPLETE',expected_rows=len(registered),rows=rows,
        total_steps=sum(r['steps'] for r in rows),
        total_head_warmup_steps=sum(r['warmup']['summary']['optimizer_steps'] for r in rows),
        total_calibration_readouts=sum(r['calibration']['extra_source_readouts'] for r in rows),
        incremental_collection=dict(reused_closed_rows=list(reused),newly_read_closed_rows=list(new),
            previous_audit_read_at=previous['read_at'],fresh_audit_read_at=fresh['read_at'],
            immutable_rows_reused=True,live_gzip_opened=False,target_access=False))
    # Original expanded arrays remain present and unchanged for every cached row.
    assert all(combined[rid]==row for rid,row in reused.items())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=PACKAGES,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    value=collect(a.kind,a.output);a.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    change=value.get('incremental_collection',{})
    print(json.dumps(dict(status=value['status'],rows=len(value['rows']),steps=value['total_steps'],
        reused=len(change.get('reused_closed_rows',[])),new=len(change.get('newly_read_closed_rows',[])),errors=value['errors'])))
