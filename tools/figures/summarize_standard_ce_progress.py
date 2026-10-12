"""Snapshot all completed clean scores with exact per-network seed counts."""
import argparse
import csv
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import statistics

NAMES={'residual_fusion':'CVS','resnet18':'ResNet-18-1D','resnet34':'ResNet-34-1D','resnet50':'ResNet-50-1D',
 'vgg16_bn':'VGG-16-BN-1D','densenet121':'DenseNet-121-1D','convnext_tiny':'ConvNeXt-Tiny-1D',
 'transformer6':'Transformer (6 layers)','bilstm3':'BiLSTM (3 layers)',
 'cvresnet18_1d':'Complex ResNet-18-1D','cvresnet18_2d':'Complex ResNet-18-2D',
 'resnet18_2d':'ResNet-18-2D','resnet50_2d':'ResNet-50-2D'}

def mean(a):return statistics.mean(a)
def sd(a):return statistics.stdev(a) if len(a)>1 else None
def csvwrite(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(snapshot,output):
    if output.exists():raise FileExistsError('Preserve existing export snapshot')
    output.mkdir(parents=True)
    rows=snapshot['rows'];keys=set();flat=[];resources=[]
    for r in rows:
        v=r['row']['variant'];seed=r['row']['model_seed'];key=(v,seed)
        if v not in NAMES or key in keys:raise ValueError('Excluded network or duplicate seed')
        keys.add(key)
        results=r['scores']['results']
        if len(results)!=14:raise ValueError('Incomplete strata')
        for s in results:
            cm=s['confusion'];count=sum(map(sum,cm));tp=[cm[i][i] for i in range(6)]
            precision=[tp[i]/sum(row[i] for row in cm) if sum(row[i] for row in cm) else 0 for i in range(6)]
            recall=[tp[i]/sum(cm[i]) if sum(cm[i]) else 0 for i in range(6)]
            f1=mean([2*p*q/(p+q) if p+q else 0 for p,q in zip(precision,recall)])
            if count!=s['query_count'] or abs(sum(tp)/count-s['accuracy'])>1e-12 or abs(f1-s['macro_f1'])>1e-12:raise ValueError('Score/confusion mismatch')
            flat.append(dict(row_id=s['row_id'],variant=v,network=NAMES[v],model_seed=seed,view='clean',
                dimension=s['dimension'],stratum=s['stratum'],query_count=count,accuracy_pct=100*s['accuracy'],macro_f1_pct=100*s['macro_f1']))
        profile=r['resource_profile'];pred=r.get('prediction_complete',{})
        resources.append(dict(variant=v,network=NAMES[v],model_seed=seed,hardware=profile['hardware'],
            **{k:profile[k] for k in ('total_parameters','trainable_parameters','gradient_used_parameters','resident_state_bytes','inference_batch1_ms','inference_batch128_ms','training_batch128_ms','conv_linear_macs_per_sample','fft_calls_per_sample')},
            training_elapsed_seconds=r['completion'].get('elapsed_seconds'),
            training_peak_cuda_allocated_bytes=r.get('training_peak_cuda_allocated_bytes'),
            full_clean_inference_seconds=pred.get('inference_seconds'),full_clean_peak_cuda_allocated_bytes=pred.get('peak_cuda_allocated_bytes')))
    metric={(r['row']['variant'],r['row']['model_seed'],s['dimension'],s['stratum']):s for r in rows for s in r['scores']['results']}
    summary=[];rx=[];tx=[];paired=[]
    for v in NAMES:
        group=[r for r in rows if r['row']['variant']==v];seeds=sorted(r['row']['model_seed'] for r in group)
        if not seeds:raise ValueError('No scored seeds for '+v)
        a=[100*metric[v,s,'overall','ALL']['accuracy'] for s in seeds];f=[100*metric[v,s,'overall','ALL']['macro_f1'] for s in seeds]
        w=[100*min(s['accuracy'] for s in r['scores']['results'] if s['dimension']=='receiver') for r in group]
        summary.append(dict(variant=v,network=NAMES[v],total_parameters=group[0]['resource_profile']['total_parameters'],
            actual_seed_count=len(seeds),expected_seed_count=8,original_completed_seeds=sum(s<=2026092704 for s in seeds),
            new_completed_seeds=sum(s>=2026092705 for s in seeds),completed_model_seeds=';'.join(map(str,seeds)),
            accuracy_pct=mean(a),accuracy_sd_pp=sd(a),macro_f1_pct=mean(f),macro_f1_sd_pp=sd(f),
            worst_receiver_accuracy_pct=mean(w),worst_receiver_sd_pp=sd(w)))
        for dim,out in [('receiver',rx),('transmitter',tx)]:
            strata=[s['stratum'] for s in group[0]['scores']['results'] if s['dimension']==dim]
            for stratum in strata:
                ar=[100*metric[v,s,dim,stratum]['accuracy'] for s in seeds];fr=[100*metric[v,s,dim,stratum]['macro_f1'] for s in seeds]
                out.append(dict(variant=v,network=NAMES[v],stratum=stratum,query_count_per_seed=metric[v,seeds[0],dim,stratum]['query_count'],
                    actual_seed_count=len(seeds),accuracy_pct=mean(ar),accuracy_sd_pp=sd(ar),macro_f1_pct=mean(fr),macro_f1_sd_pp=sd(fr)))
        if v!='residual_fusion':
            for s in seeds:
                if ('residual_fusion',s,'overall','ALL') not in metric:continue
                paired.append(dict(control='residual_fusion',baseline=v,model_seed=s,
                    accuracy_difference_pp=100*(metric['residual_fusion',s,'overall','ALL']['accuracy']-metric[v,s,'overall','ALL']['accuracy']),
                    macro_f1_difference_pp=100*(metric['residual_fusion',s,'overall','ALL']['macro_f1']-metric[v,s,'overall','ALL']['macro_f1'])))
    tables={'network_summary.csv':summary,'per_seed_metrics.csv':flat,'receiver_summary.csv':rx,'transmitter_summary.csv':tx,
        'paired_differences.csv':paired,'resource_profiles.csv':resources}
    for filename,data in tables.items():csvwrite(output/filename,data)
    classes=rows[0]['classes']
    if any(r['classes']!=classes for r in rows):raise ValueError('Class map changed')
    local=datetime.fromisoformat(snapshot['read_at_utc']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S HKT')
    raw=dict(run_id='20261012-phase1-standard-ce-seed-extension-manysig-m52-r01',status='VERIFIED',
        snapshot_hkt=local,actual_scored_models=len(rows),network_count=13,classes=classes,
        progress=snapshot['progress'],run_states=snapshot['states'],
        rows=[dict(row=r['row'],scores=r['scores']) for r in rows])
    (output/'verified_results.json').write_text(json.dumps(raw,ensure_ascii=False),encoding='utf-8')
    (output/'excel_input.json').write_text(json.dumps(dict(raw=raw,tables=tables),ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(snapshot_hkt=local,models=len(rows),seed_counts={r['network']:r['actual_seed_count'] for r in summary}),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    summarize(json.loads(a.snapshot.read_text(encoding='utf-8')),a.output)
