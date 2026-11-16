"""Describe fixed top/bottom CE-proxy comparisons for source-only screening."""
import argparse
import json
import math
from pathlib import Path

GROUPS=('t2','cls_head','other')


def analyze(result):
    if (result['status']!='SOURCE_CE_HEAD_PROXY_SCREEN_COMPLETE'
            or result['U_truth_reads'] or result['V_reads'] or result['target_reads']
            or result['original_parameter_updates'] or result['new_training']
            or result['fixed_offsets']!=list(range(8))):
        raise ValueError('Only complete fixed source-only diagnostic output is admissible')
    settings=result['proxy_screen']
    if set(settings)!={'ten_class','source_rx_6'}:raise ValueError('Both registered settings required')
    findings={}
    for setting,data in settings.items():
        pairs=data['cross_batch_pairs'];rows=data['per_L_extremes']
        if len(pairs)!=56 or len(rows)!=8 or {r['labeled_offset'] for r in rows}!=set(range(8)):
            raise ValueError('Every preregistered cross-batch comparison required')
        expected={(i,j) for i in range(8) for j in range(8) if i!=j}
        if {(p['labeled_offset'],p['unlabeled_offset']) for p in pairs}!=expected:
            raise ValueError('Distinct cross-batch source pairs required')
        groups={}
        for group in GROUPS:
            valid=[]
            for row in rows:
                high=row['highest']['actual_full_L_vs_raw_U'][group]
                low=row['lowest']['actual_full_L_vs_raw_U'][group]
                values=(high['cosine'],low['cosine'],high['dot'],low['dot'],high['right_norm'],low['right_norm'])
                if any(v is None for v in values):continue
                if not all(math.isfinite(v) for v in values):raise ValueError('Finite source relations required')
                valid.append(values)
            if not valid:
                groups[group]=dict(defined_pairs=0);continue
            def mean(i):return sum(v[i] for v in valid)/len(valid)
            groups[group]=dict(defined_pairs=len(valid),highest_cosine_mean=mean(0),lowest_cosine_mean=mean(1),
                paired_cosine_difference_mean=mean(0)-mean(1),
                highest_more_aligned=sum(v[0]>v[1]+1e-6 for v in valid),
                lowest_more_aligned=sum(v[1]>v[0]+1e-6 for v in valid),
                ties=sum(abs(v[0]-v[1])<=1e-6 for v in valid),
                highest_conflicts=sum(v[0]<-1e-6 for v in valid),lowest_conflicts=sum(v[1]<-1e-6 for v in valid),
                highest_dot_mean=mean(2),lowest_dot_mean=mean(3),
                highest_U_norm_mean=mean(4),lowest_U_norm_mean=mean(5))
        findings[setting]=dict(groups=groups,source_L_batches=len(rows),ordered_cross_pairs=len(pairs),
            proxy_high_mean=sum(r['highest']['proxy_CE_head']['cosine'] for r in rows)/len(rows),
            proxy_low_mean=sum(r['lowest']['proxy_CE_head']['cosine'] for r in rows)/len(rows))
    return dict(status='ANALYZED_SOURCE_CE_PROXY_SCREEN',settings=findings,
        interpretation='Dependent pairs, one previously exposed seed, same E20 parameters. '
        'This screening does not estimate regrouped contrast/MixStyle gradients, future AdamW updates, '
        'coverage, training effects or target performance.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');a=p.parse_args()
    print(json.dumps(analyze(json.loads(Path(a.input).read_text(encoding='utf-8-sig'))),ensure_ascii=False,allow_nan=False))
