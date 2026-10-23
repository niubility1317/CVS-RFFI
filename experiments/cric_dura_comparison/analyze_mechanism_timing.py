"""Audit the two completed adjunct timing panels; never measures a model."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from .analyze_main import write_csv
from .analyze_timing import percentile
from .reproduce_final import PortableInput


def run(folder, output):
    source=PortableInput(folder);manifest=source.read('manifest.json')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);proof={}
    for panel,case_count in (('fine_timing',36),('structure_timing',18)):
        c=source.read(manifest['analysis_configs'][panel]);root=Path(c['output_root'])
        done=source.read(root/'complete.json');resolved=source.read(root/'resolved.json')
        summary=source.read(root/'summary.json');profiles=source.read(root/'operator_profiles.json')
        cases=done['cases'];batches=[1,8,32,64,128]
        if (done['status']!='TIMING_COMPLETE' or not done['exclusive_GPU'] or not done['source_IQ_only']
                or len(cases)!=case_count or len(set(cases))!=case_count or done['batches']!=batches or done['rounds']!=5
                or resolved['cases']!=cases or resolved['precision']!='FP32' or resolved['TF32']
                or resolved['restored_IQ_cache'] or not resolved['source_IQ_only'] or c['external_models']):
            raise ValueError('Adjunct timing contract differs')
        if panel=='structure_timing' and (not c['mechanisms_only'] or c['G']['roles']!='L'):
            raise ValueError('Strict structure timing is not L-only')
        rows=[json.loads(line) for line in (root/'samples.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        expected={(name,b,r) for name in cases for b in batches for r in range(5)}
        keys={(r['case'],r['batch'],r['round']):r for r in rows}
        reported={(r['case'],r['batch'],r['round']):r for r in summary['results']}
        if len(keys)!=len(rows) or set(keys)!=expected or set(reported)!=expected or not summary['exclusive_GPU']:
            raise ValueError('Missing/duplicate timing cell')
        grouped=defaultdict(list);rounds=[]
        for key,r in keys.items():
            repeats=1000 if r['batch']==1 else 100
            if (r['repeats']!=repeats or not r['exclusive_GPU'] or r['warmup']!=(100 if repeats==1000 else 20)
                    or len(r['wall_ms'])!=repeats or len(r['device_ms'])!=repeats or len(r['H2D_D2H_ms'])!=20
                    or r['incremental_peak_bytes']!=max(0,r['peak_allocated_bytes']-r['resident_baseline_bytes'])):
                raise ValueError('Invalid timing samples/memory accounting')
            if any(not math.isfinite(x) or x<0 for k in ('wall_ms','device_ms','H2D_D2H_ms') for x in r[k]):
                raise ValueError('Non-finite timing')
            median=statistics.median(r['wall_ms']);p95=percentile(r['wall_ms'],.95)
            for field,value in (('median_ms',median),('P95_ms',p95),('throughput_IQ_s',r['batch']*1000/median)):
                if not math.isclose(reported[key][field],value,rel_tol=1e-10,abs_tol=1e-10):
                    raise ValueError('Round summary differs from raw calls')
            grouped[key[:2]].append(r)
            rounds.append(dict(case=key[0],batch=key[1],round=key[2],median_ms=median,P95_ms=p95))
        def route(name):
            if name.startswith('T'):return name.rsplit('_',1)[0]
            if name=='F_graph':return 'F CUDA graph'
            if name=='F_eager' or name.endswith('_F'):return 'F eager'
            if any(name.startswith(p['name']+'_') for p in c['G_mechanisms']):return 'G variant eager; F eager if GF'
            return 'G graph+residual reuse; F graph'
        result=[]
        for case in cases:
            for batch in batches:
                group=sorted(grouped[case,batch],key=lambda r:r['round'])
                pool=lambda field:[x for r in group for x in r[field]]
                wall=pool('wall_ms');transfer=pool('H2D_D2H_ms');medians=[statistics.median(r['wall_ms']) for r in group]
                result.append(dict(case=case,execution=route(case),batch=batch,rounds=5,wall_calls=len(wall),
                    median_ms=statistics.median(wall),P95_ms=percentile(wall,.95),device_median_ms=statistics.median(pool('device_ms')),
                    CPU_return_median_ms=statistics.median(transfer),CPU_return_P95_ms=percentile(transfer,.95),
                    round_median_SD_ms=statistics.stdev(medians),throughput_IQ_s=batch*1000/statistics.median(wall),
                    first_call_ms=group[0]['cold_first_call_ms'],resident_baseline_min_bytes=min(r['resident_baseline_bytes'] for r in group),
                    resident_baseline_max_bytes=max(r['resident_baseline_bytes'] for r in group),incremental_peak_max_bytes=max(r['incremental_peak_bytes'] for r in group)))
        write_csv(out/(panel+'_summary.csv'),result);write_csv(out/(panel+'_rounds.csv'),rounds)
        proof[panel]=dict(status='VERIFIED',run_id=c['run_id'],cases=case_count,case_rounds=len(rows),
            GPU_wall_calls=sum(len(r['wall_ms']) for r in rows),CPU_return_calls=sum(len(r['H2D_D2H_ms']) for r in rows),
            precision='FP32',TF32=False,exclusive_GPU=True,source_IQ_only=True,
            F_parameters=resolved['F_parameters'],G_parameters=resolved['G_parameters'],mechanism_parameters=resolved['mechanism_parameters'],
            scheduler_reservation_bytes=resolved['scheduler_reservation_bytes'],operator_profiles=profiles,
            limitations=['Five timing rounds of fixed seed01 weights, not five model seeds.',
                'Absolute resident VRAM includes all models and graph pools, not a single-model measurement.',
                'F-only mechanism calls are eager; F-mechanism GF uses graphs and residual reuse.',
                'G-mechanism variants are eager; no optimized-route speedup is inferred from their values.',
                'First call follows loading/profiling, not a fresh process; CPU return includes transfer and output conversion.',
                'The delivered uniform latency-r02 remains the official default result; it is not remeasured here.'])
    (out/'mechanism_timing_verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{n:v[n] for n in ('status','cases','case_rounds','GPU_wall_calls','CPU_return_calls')} for k,v in proof.items()}))
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.input,a.output)
