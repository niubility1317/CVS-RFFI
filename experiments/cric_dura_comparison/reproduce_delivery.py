"""Reproduce only the newly closed panels in the final immutable delivery."""
import argparse,json
from pathlib import Path
from .reproduce_final import run as replay,compare_csv
from .analyze_structure import run as structure
from .analyze_mechanism_timing import run as timing
from .final_external import run as external
from .final_costs import run as costs


def run(package,output):
    package=Path(package).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    inputs=package/'final_inputs';proof={}
    replay(inputs,out/'replay');structure(inputs,out/'structure');timing(inputs,out/'timing');external(inputs,out/'external-final')
    costs(package/'final_cost_inputs',out/'cost_ledger')
    mapping={'structure':package/'final_results/structure','timing':package/'final_results/timing',
        'external-final':package/'final_results/external-final','cost_ledger':package/'final_results/cost_ledger',
        'replay/external':package/'final_results/external','replay/gfactor':package/'final_results/gfactor'}
    for name,expected in mapping.items():
        proof[name]={p.name:compare_csv(p,out/name/p.name) for p in sorted(expected.glob('*.csv'))}
        if not proof[name]:raise ValueError('No expected CSV')
    result=dict(status='UNPACKED_REPRODUCED',panels=proof,
        numeric_cells=sum(r['numeric_cells'] for group in proof.values() for r in group.values()),
        maximum_absolute_difference=max(r['maximum_absolute_difference'] for group in proof.values() for r in group.values()),
        previous_verified_panels_not_rerun=True,statistics_stdlib_only=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','numeric_cells','maximum_absolute_difference')}));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.package,a.output)
