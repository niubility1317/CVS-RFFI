"""Summarize existing independently scored paired results; no new predictions."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

SEEDS = (2026092701, 2026092702, 2026092703)
VIEWS = ('clean','practical_high','practical_mid','practical_low_suburban',
         'practical_high_urban','practical_mid_urban','practical_low_urban')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    if data['status'] != 'PAIRED_COMPLETE' or data['truth_last'] is not True or tuple(data['seeds']) != SEEDS:
        raise ValueError('Completed same-seed truth-last comparison required')
    rows = [r for r in data['rows'] if r['readout'] == 'DURA-LU' and r['task'] == 'combined_target']
    for r in rows:
        if r['count'] <= 0 or r['net'] != r['fixed'] - r['broken'] or abs(r['delta_pp'] - 100*r['net']/r['count']) > 1e-10:
            raise ValueError('Inconsistent same-packet comparison')
    directory = args.output
    directory.mkdir(parents=True, exist_ok=False)
    with (directory/'dura_per_seed_strata.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    overall=[]
    for baseline in ('Tri-SIG-TIME','PL'):
        for view in VIEWS:
            selected=[r for r in rows if r['baseline']==baseline and r['view']==view and r['dimension']=='overall']
            if len(selected)!=3 or {r['seed'] for r in selected}!=set(SEEDS) or any(r['count']!=168000 for r in selected):
                raise ValueError('Three complete matched seeds required')
            selected.sort(key=lambda r:r['seed'])
            differences=[r['delta_pp'] for r in selected]
            overall.append(dict(baseline=baseline,view=view,candidate_mean=mean(r['candidate_accuracy'] for r in selected),
                baseline_mean=mean(r['baseline_accuracy'] for r in selected),delta_pp=mean(differences),
                delta_sd_pp=stdev(differences),per_seed_delta_pp=differences,
                fixed=sum(r['fixed'] for r in selected),broken=sum(r['broken'] for r in selected)))
    (directory/'dura_overall.json').write_text(json.dumps(overall,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(overall,indent=2))


if __name__=='__main__':main()
