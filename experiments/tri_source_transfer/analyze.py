"""Aggregate the complete fixed source diagnostic; never select an arm."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as stats

PARTS=('pseudo_CE','pair_KL','shallow','joint','extrap_delta')


def summarize(folder):
    folder=Path(folder)
    done=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    rows=[json.loads(s) for s in (folder/'observations.jsonl').read_text(encoding='utf-8').splitlines()]
    keys={(r['setting'],r['source_epoch'],r['held_rx'],r['reference_mode']) for r in rows}
    if done['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or len(rows)!=38 or len(keys)!=38:
        raise ValueError('A closed complete 38-cell diagnostic is required')
    buckets=defaultdict(list)
    for r in rows:
        if (r['U_truth_reads'] or r['V_reads'] or r['target_reads'] or r['checkpoint_parameter_writes']
                or not r['physical_reference_update_disjoint'] or len(r['arms'])!=8):
            raise ValueError('Role or comparison matrix differs')
        buckets[(r['setting'],r['source_epoch'],r['reference_mode'])].append(r)
    summary=[]
    first_order_agreement={k:dict(agree=0,opposite=0,near_zero=0) for k in ('full_U',*PARTS)}
    for r in rows:
        for key in first_order_agreement:
            if key=='full_U':
                observed=r['arms']['full_U']['benefit_vs_L_only']
                predicted=r['arms']['full_U']['reference_first_order']-r['arms']['L_only']['reference_first_order']
            else:
                observed=r['arms']['without_'+key]['benefit_of_included_component']
                predicted=r['arms']['full_U']['reference_first_order']-r['arms']['without_'+key]['reference_first_order']
            group=first_order_agreement[key]
            if abs(observed)<=1e-6 or abs(predicted)<=1e-6:group['near_zero']+=1
            elif observed*predicted>0:group['agree']+=1
            else:group['opposite']+=1
    for (setting,epoch,mode), values in buckets.items():
        item=dict(setting=setting,epoch=epoch,mode=mode,n_rx=len(values),
                  mean_reference_ce=stats.mean(x['reference']['ce'] for x in values),
                  full_U_benefit=stats.mean(x['arms']['full_U']['benefit_vs_L_only'] for x in values),
                  PL_benefit=stats.mean(x['arms']['ordinary_PL_at_Tri']['benefit_vs_L_only'] for x in values),
                  full_U_positive=sum(x['arms']['full_U']['benefit_vs_L_only']>1e-6 for x in values),
                  accepted_mean=stats.mean(x['actual_U_accepted']/x['U_n'] for x in values),components={})
        for name in PARTS:
            benefits=[x['arms']['without_'+name]['benefit_of_included_component'] for x in values]
            losses=[x['component_losses'][name] for x in values]
            item['components'][name]=dict(mean=stats.mean(benefits),minimum=min(benefits),maximum=max(benefits),
                 positive=sum(b>1e-6 for b in benefits),negative=sum(b< -1e-6 for b in benefits),
                 near_zero=sum(abs(b)<=1e-6 for b in benefits),mean_loss=stats.mean(losses),
                 by_rx={str(x['held_rx']):b for x,b in zip(values,benefits)})
        summary.append(item)
    costs=[]
    for r in rows:
        t=r['timings'];base=t['native_graph_and_gradient_seconds']+r['arms']['full_U']['optimizer_seconds']
        extra=t['component_gradient_seconds']+t['reference_gradient_seconds']+t['plain_PL_gradient_seconds']
        extra+=sum(v['optimizer_seconds'] for k,v in r['arms'].items() if k!='full_U')
        extra+=sum(v['reference_forward_seconds'] for v in r['arms'].values())
        costs.append(dict(setting=r['setting'],epoch=r['source_epoch'],rx=r['held_rx'],
                          native_instrumented_seconds=base,extra_measured_seconds=extra,
                          extra_ratio=extra/base,total_snapshot_seconds=t['total_snapshot_seconds']))
    cost_summary={k:dict(mean=stats.mean(x[k] for x in costs),median=stats.median(x[k] for x in costs),
                        minimum=min(x[k] for x in costs),maximum=max(x[k] for x in costs))
                  for k in ('native_instrumented_seconds','extra_measured_seconds','extra_ratio','total_snapshot_seconds')}
    return dict(status='ANALYZED',snapshots=len(rows),conditions=summary,costs=cost_summary,
                actual_update_first_order_vs_measured=first_order_agreement,
                sum_snapshot_seconds=sum(x['total_snapshot_seconds'] for x in costs),
                threshold_note='1e-6 CE is a reporting bin only, not significance or selection threshold',
                cost_note='CPU2 FP32, instrumented native estimate; extra excludes construction/source load and some observer work; no GPU cost inference')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args()
    data=summarize(a.folder)
    Path(a.folder,'summary.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    for row in data['conditions']:
        print(json.dumps(row))
    print(json.dumps({'costs':data['costs'],'sum_snapshot_seconds':data['sum_snapshot_seconds']}))
