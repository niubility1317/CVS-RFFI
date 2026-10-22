"""Portable, stdlib-only reconstruction of the completed core factorial."""
import argparse
import csv
import json
import math
from pathlib import Path

from .analyze_core import run as aggregate
from .core_results import run as compact

FILES=('core_per_seed.csv','core_summary.csv','core_paired_effects.csv',
       'core_net_corrections.csv','core_F_cost.csv')
COMPACT_FILES=('core_overall_summary.csv','core_overall_effects.csv',
    'core_restoration_paired.csv','core_completed_F_cost.csv',
    'core_source_geometry.csv','core_source_interventions.csv')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def compare(left,right):
    with Path(left).open(encoding='utf-8',newline='') as f:a=list(csv.DictReader(f))
    with Path(right).open(encoding='utf-8',newline='') as f:b=list(csv.DictReader(f))
    if len(a)!=len(b):raise ValueError('CSV row coverage differs: '+Path(left).name)
    count=0;maximum=0.
    for x,y in zip(a,b):
        if set(x)!=set(y):raise ValueError('CSV columns differ')
        for key in x:
            try:old,new=float(x[key]),float(y[key])
            except ValueError:
                if x[key]!=y[key]:raise ValueError('CSV label/identity differs: '+key)
                continue
            if not (math.isfinite(old) and math.isfinite(new)) or abs(old-new)>1e-10:
                raise ValueError('CSV numeric value differs beyond 1e-10: '+key)
            count+=1;maximum=max(maximum,abs(old-new))
    return dict(rows=len(a),numeric_cells=count,maximum_absolute_difference=maximum)


def run(folder,output,expected=None,figures=None):
    folder=Path(folder).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a fresh reproduction output directory')
    mapping=read(folder/'path_mapping.json')
    def remap(value):
        if isinstance(value,dict):return {k:remap(v) for k,v in value.items()}
        if isinstance(value,list):return [remap(v) for v in value]
        return str(folder/mapping[value]) if isinstance(value,str) and value in mapping else value
    c=remap(read(folder/'original_config.json'))
    # An original-T1 reference has no single synthetic completion file. Its
    # actual three-segment costs are reconstructed by core_results instead.
    for cell in c['cells']:
        source=cell['source'];path=source['completion']
        if path not in [str(folder/v) for v in mapping.values()]:
            if cell['bits']!='111' or source['new_F']:
                raise ValueError('Missing frozen source metadata for an actual trained F')
            source['completion']=str(output/'absent_original_T1_reference.json')
    output.mkdir(parents=True)
    c['output_root']=str(output/'recalculated')
    aggregate(c)
    comparisons={name:compare(folder/'remote_analysis'/name,Path(c['output_root'])/name) for name in FILES}
    compact(folder,output/'compact',figures)
    compact_comparisons={}
    if expected:
        compact_comparisons={name:compare(Path(expected)/name,output/'compact'/name) for name in COMPACT_FILES}
        if read(Path(expected)/'core_cost_summary.json')!=read(output/'compact/core_cost_summary.json'):
            raise ValueError('Completed F cost/source diagnostic summary differs')
    verified=read(Path(c['output_root'])/'verification.json')
    verified.update(portable_reproduction=True,all_five_CSVs_equivalent=comparisons,
        compact_tables_equivalent=compact_comparisons,cross_Python_absolute_tolerance=1e-10,
        numeric_cells_compared=sum(x['numeric_cells'] for x in comparisons.values()),
        maximum_absolute_difference=max(x['maximum_absolute_difference'] for x in comparisons.values()),
        model_loading=False,IQ_loading=False,target_truth_reads=0)
    (output/'core_reproduction_verification.json').write_text(json.dumps(verified,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED',logical_cells=24,serialized_CM_recomputed=verified['serialized_CM_recomputed'],
        numeric_cells_compared=verified['numeric_cells_compared'],compact_tables=len(compact_comparisons))))
    return verified


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--expected');p.add_argument('--figures')
    a=p.parse_args();run(a.input,a.output,a.expected,a.figures)
