"""Read runtime evidence for completed physical-route model seeds."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True,type=Path);root=p.parse_args().project/'runs'
    records=[]
    for seed in (2026092701,2026092702,2026092703):
        run=('20261110-phase1-pamf-ptbr-route-f0-s2026092701-m1-r01' if seed==2026092701 else
             '20261110-phase1-pamf-ptbr-route-f0-s2026092702-03-m2-r01')
        path=root/run/'F'/f'ptbr-route-s{seed}'/'source'/'complete.json'
        source=json.loads(path.read_text())
        if source['status']!='SOURCE_FROZEN' or source['epoch']!=50 or source['updates']!=11100:
            raise ValueError('Completed fixed source trajectory required')
        records.append(dict(seed=seed,seconds=source.get('seconds'),
            cumulative_training_seconds=source.get('cumulative_training_seconds'),
            reused_prefix_seconds=source.get('reused_prefix_seconds'),
            segment_seconds=[s.get('seconds') for s in source.get('segments',[])]))
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
