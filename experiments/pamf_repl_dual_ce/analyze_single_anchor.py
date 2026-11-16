"""Compare fixed dual versus single-anchor runs on matched clean IDs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from experiments.pamf_repl_weight.analyze_paired import read, prediction, compare


def analyze(dual_root,single_w6_root,single_w3_root):
    roots=[Path(v) for v in (dual_root,single_w6_root,single_w3_root)]
    if any(read(p/'pipeline_state.json')['status']!='ANALYZED' for p in roots):
        raise ValueError('All fixed runs must complete independent scoring')
    dual,six,three=[read(p/'sources.json') for p in roots]
    runtime=read(roots[0].parents[1]/'releases'/(roots[0].name+'-release')/'runtime.json')
    context=read(runtime['context']);rows=[]
    for setting in ('ten_class','source_rx_6'):
        for weight,baselines,recipe in ((6,six,'step_guard'),(3,three,'step_weight3')):
            candidate_id=f'{setting}-repl-dual_ce{weight}-s2026092701'
            baseline_id=f'{setting}-repl-{recipe}-s2026092701'
            candidate,baseline=dual[candidate_id],baselines[baseline_id]
            cfgs=[];scored_correct=[]
            for rid,entry in ((candidate_id,candidate),(baseline_id,baseline)):
                done=read(Path(entry['source_output'])/'complete.json');cfg=done['config']
                if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100
                        or done['target_access'] or done['U_truth_read'] or cfg['row_id']!=rid
                        or cfg['model_seed']!=2026092701):
                    raise ValueError('Matched frozen source-only candidate required')
                score=read(Path(entry['scores'])/'scores.json')
                if read(Path(entry['scores'])/'complete.json')['status']!='SCORED_COMPLETE' or not score['truth_last']:
                    raise ValueError('Independent truth-last clean score required')
                values=[x for x in score['results'] if x['test'] in ('cross_rx','cross_both')
                        and x['readout']=='raw' and x['view']=='clean' and x['dimension']=='overall']
                if len(values)!=2 or {x['test'] for x in values}!={'cross_rx','cross_both'}:
                    raise ValueError('Both main clean test scores required')
                scored_correct.append(sum(sum(row[i] for i,row in enumerate(x['confusion'])) for x in values))
                cfgs.append(cfg)
            if (cfgs[0]['protocol_id']!=cfgs[1]['protocol_id'] or
                    cfgs[0]['source_contract']!=cfgs[1]['source_contract']):
                raise ValueError('Same source contract required')
            total=Counter();per_test={}
            for task in ('cross_rx','cross_both'):
                ids,new=prediction(candidate['predictions'],task);old_ids,old=prediction(baseline['predictions'],task)
                if not np.array_equal(ids,old_ids):raise ValueError('Exact physical IDs required')
                truth=read(context['handles'][cfgs[0]['protocol_id']]['tests'][task]['truth'])
                per_test[task]=compare(ids,new,old,truth);total.update(per_test[task]['overall'])
            if (total['candidate_correct'],total['baseline_correct'])!=tuple(scored_correct):
                raise ValueError('Independent score and paired predictions disagree')
            rows.append(dict(setting=setting,weight=weight,candidate=candidate_id,baseline=baseline_id,
                             combined=dict(total),delta_pp=100*total['net']/total['count'],per_test=per_test))
    return dict(status='ANALYZED_DUAL_VS_SINGLE_ANCHOR',rows=rows,
        interpretation='Fixed same-weight exploratory comparisons, exposed targets and one seed. '
        'Different parameter and minibatch trajectories; target and U truth do not select models.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('dual-root','single-w6-root','single-w3-root'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(analyze(a.dual_root,a.single_w6_root,a.single_w3_root),ensure_ascii=False,allow_nan=False))
