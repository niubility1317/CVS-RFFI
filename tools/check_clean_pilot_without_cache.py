"""Read-only configuration check, excluding the known absent restored bank."""
import json
import argparse
from pathlib import Path
import tempfile

from experiments.pamf_phase2_practical.clean_scheduler import validate_plan
from experiments.pamf_phase2_practical.io import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    path=Path(args.config)
    config=read(path);selected=[dict(j) for j in config['jobs'] if j['group']=='pamf']
    with tempfile.TemporaryDirectory(prefix='clean-plan-check-') as folder:
        for i,job in enumerate(selected):
            rt=read(job['runtime']);restored=Path(rt.pop('restored_root'))
            if restored.exists():raise ValueError('Cache status changed; inspect before using this check')
            p=Path(folder)/f'rt{i}.json';p.write_text(json.dumps(rt));job['runtime']=str(p)
        result=validate_plan(dict(config,jobs=selected))
    print(json.dumps(dict(status='PAMF_INPUT_PATHS_PRESENT_CACHE_MISSING',jobs=result['jobs'],
                          cache_ready=False,query_read=False,formal_experiment_started=False)))


if __name__=='__main__':main()
