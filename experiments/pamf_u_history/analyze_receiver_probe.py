"""Complete evidence arithmetic for the real source probe, without new inference."""
import argparse
import json
import math
from pathlib import Path
import statistics
from .analyze import records,dump_csv
from .receiver_probe import RUN,RECIPE


def analyze(evidence,output):
    root=Path(evidence);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    result=json.loads((root/'result.json').read_text(encoding='utf-8'));smoke=json.loads((root/'smoke.json').read_text(encoding='utf-8'))
    rows,_=records(root/'gradient_steps.jsonl',complete=True)
    if (result['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or result['run_id']!=RUN or result['recipe']!=RECIPE
        or smoke['status']!='PASS' or len(rows)!=9 or rows!=result['gradient_rows']
        or [r['step'] for r in rows]!=list(range(1,10)) or result['signature_views']!=1
        or result['signature_parameters']!=18976 or result['L_seen']!=1152 or result['U_seen']!=2304
        or result['V_seen']!=27000 or not result['all_parameters_and_registered_buffers_unchanged']
        or any(result[k] for k in ('optimizer_updates','U_truth_read','target_reads','G_updates'))):
        raise ValueError('Complete legal real source probe required')
    same_values=True;summary=[]
    for row in rows:
        same_values&=all(abs(row['gradients']['U'][t]['loss']-row['gradients']['LU'][t]['loss'])<=1e-6 for t in ('instance','relation'))
        for scope in ('U','LU'):
            for component in ('instance','relation'):
                r=row['gradients'][scope][component]
                if not all(math.isfinite(r[k]) for k in ('loss','norm','base_norm','dot','cosine')) or r['norm']<=0:
                    raise ValueError('Finite real gradients at every batch required')
    if not same_values:raise ValueError('Scope comparison changed scalar objective')
    for scope in ('U','LU'):
        for component in ('instance','relation'):
            values=[r['gradients'][scope][component] for r in rows]
            summary.append(dict(scope=scope,component=component,batches=9,
                loss_median=statistics.median(v['loss'] for v in values),norm_median=statistics.median(v['norm'] for v in values),
                base_norm_median=statistics.median(v['base_norm'] for v in values),
                cosine_median=statistics.median(v['cosine'] for v in values),negative_dot_batches=sum(v['dot']<0 for v in values)))
    quality=result['quality'];byrx=result['extra_hard_by_RX'];expected_pass=quality['extra_hard']['n']>=200 and all(r['n']>=20 and r['precision_percent']>=99.5 for r in byrx.values())
    if expected_pass!=result['route_source_necessary_pass']:raise ValueError('Source route necessary rule recount mismatch')
    expected={}
    for component in ('instance','relation'):
        base=next(v['base_norm_median'] for v in summary if v['component']==component and v['scope']=='U')
        norm=max(v['norm_median'] for v in summary if v['component']==component)
        expected[component]=min(.25,.1*base/norm)
    if expected!=result['gradient_scale_suggestion']:raise ValueError('Fixed source norm rule mismatch')
    aggregate={k:result[k] for k in ('status','run_id','code_commit','recipe','actual_checkpoint_epoch','actual_checkpoint_updates',
        'signature_views','signature_parameters','L_seen','U_seen','V_seen','quality','extra_hard_by_RX','gradient_scale_suggestion',
        'route_source_necessary_pass','all_parameters_and_registered_buffers_unchanged','optimizer_updates','U_truth_read','target_reads','G_updates','seconds')}
    aggregate.update(parsed_step_records=9,identical_scalar_objectives=same_values,gradient_summary=summary,
        scope='fixed E20 source gradient and route diagnostic;no F/G training,causal efficacy or target claim')
    (out/'diagnostic_evidence.json').write_text(json.dumps(aggregate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    dump_csv(out/'gradient_summary.csv',summary)
    dump_csv(out/'source_route_by_RX.csv',[dict(RX=rx,**r) for rx,r in byrx.items()])
    print(json.dumps(dict(status=aggregate['status'],source_route_pass=expected_pass,quality=quality,common_coefficients=expected,parsed_step_records=9)))
    return aggregate


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.evidence,a.output)
