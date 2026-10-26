"""Complete immutable source-mask-fit evidence; scientific failure is retained."""
import argparse
import json
from pathlib import Path
import re
from .analyze import records,dump_csv
from .covariance_fit import RUN,RECIPE,LEVELS


def analyze(evidence,output):
    root=Path(evidence);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    r=json.loads((root/'result.json').read_text(encoding='utf-8'));smoke=json.loads((root/'smoke.json').read_text(encoding='utf-8'))
    steps,_=records(root/'fit_steps.jsonl',complete=True);log=(root/'fit.log').read_text(encoding='utf-8')
    if (r['status']!='SOURCE_MASK_FIT_COMPLETE' or r['run_id']!=RUN or r['recipe']!=RECIPE or smoke['status']!='PASS'
        or r['actual_checkpoint_epoch']!=20 or r['actual_checkpoint_updates']!=4440 or r['source_L_seen']!=6300
        or r['As_physical_views']!=18900 or len(steps)!=50 or [v['step'] for v in steps]!=list(range(1,51))
        or [v['L_seen'] for v in steps]!=[min(i*128,6300) for i in range(1,51)]
        or any(v['As_seen']!=3*v['L_seen'] for v in steps) or not r['parameters_and_registered_buffers_unchanged']
        or any(r[k] for k in ('F_optimizer_updates','G_weight_reads','G_updates','U_IQ_reads','U_truth_read','V_reads','target_reads'))):
        raise ValueError('Complete original labeled-source fit required')
    fatal=re.findall(r'Traceback \(most recent call last\)|RuntimeError:|FloatingPointError:|CUDA out of memory',log)
    if fatal:raise ValueError('Fatal source stdout')
    rows=[]
    for level in LEVELS:
        d=r['diagnostics'][level]
        if d['samples']!=6300 or d['paired_views']!=18900 or d['source_cell_min']!=70 or d['source_cell_max']!=70:
            raise ValueError('Complete90 TX/RX/day cells required')
        rows.append(dict(level=level,**d))
    a={k:v for k,v in r.items() if k!='source_prefix'}
    a.update(analysis_status='ANALYZED_SOURCE_ONLY',full_step_records=50,full_stdout_bytes=len(log.encode('utf-8')),fatal_markers=fatal,
        interpretation='fixed10percent mask-budget rule failed where flagged;not a negative accuracy result or a failure of the full research direction')
    (out/'source_result_evidence.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    dump_csv(out/'source_mask_feasibility.csv',rows)
    print(json.dumps(dict(status=a['analysis_status'],all_levels_passed=r['all_levels_mask_rule_passed'],diagnostics=r['diagnostics'])))
    return a


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.evidence,a.output)
