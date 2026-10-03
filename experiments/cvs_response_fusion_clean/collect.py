"""Independently reconcile all forty-four completed rows; never load target truth."""
import argparse
import json
from pathlib import Path

from experiments.cvs_clean_eval.publish import inspect, ssh
from experiments.cvs_clean_eval.contracts import SEEDS, evaluation_variants
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_response_fusion_identity.model import VARIANTS
from experiments.cvs_response_fusion_clean.prepare import RUN, RELEASE, SOURCE_RUN, SOURCE

CLASSES = ['14-10','14-7','20-15','20-19','6-15','8-20']


def validate_terminal(data, selection, spec=None, require_classes=True):
    if (selection.get('scope')!='response_fusion_source' or selection.get('status')!='SOURCE_SELECTION_FROZEN' or
            selection.get('selected_variant') not in VARIANTS or selection.get('new_candidate_selected') is not True or
            selection.get('target_access') is not False or selection.get('target_score_used') is not False):
        raise ValueError('Frozen new source-only selection required')
    if (data.get('run_id')!=RUN or not data.get('pipeline') or data['pipeline'].get('status')!='ANALYZED' or
            data['pipeline'].get('independent_scoring_complete') is not True or 'dispatcher_process' not in data or data['dispatcher_process'] is not None):
        raise ValueError('Clean dispatcher not independently terminal')
    rows=data.get('rows',[]);expected={(v,s) for v in evaluation_variants(selection) for s in SEEDS}
    actual={(r.get('resolved',{}).get('variant'),r.get('resolved',{}).get('model_seed')) for r in rows if r.get('resolved')}
    if len(rows)!=44 or actual!=expected or len({r.get('row_id') for r in rows})!=44:
        raise ValueError('Missing/duplicate clean completion row')
    registered={r['row_id']:r for r in spec['rows']} if spec else None
    if registered is not None and set(registered)!={r['row_id'] for r in rows}:
        raise ValueError('Terminal rows differ from actual launch matrix')
    for row in rows:
        resolved=row.get('resolved') or {};done=row.get('completion') or {};provenance=row.get('provenance') or {}
        rid=resolved['variant']+'-s'+str(resolved['model_seed'])
        if ('process' not in row or row['process'] is not None or row.get('status')!='PREDICTIONS_COMPLETE' or row.get('row_id')!=rid or
                done.get('status')!='PREDICTIONS_COMPLETE' or done.get('count')!=168000 or done.get('views')!=['clean'] or
                done.get('truth_read') is not False or done.get('query_fit') is not False or
                resolved.get('truth_read') is not False or resolved.get('query_fit') is not False or resolved.get('views')!=['clean'] or
                provenance.get('status')!='VERIFIED' or provenance.get('query_fit') is not False or
                provenance.get('source_contract_roles')!='EXACT_MATCH' or provenance.get('initialization')!='SCRATCH' or
                provenance.get('ancestors')!=[] or provenance.get('selection')!='E200' or
                provenance.get('checkpoint')!=resolved.get('source_output','')+'/last.pt'):
            raise ValueError('Incomplete/contaminated prediction or live/unknown process')
        if registered is not None:
            planned=registered[rid]
            if any(resolved.get(k)!=planned.get(k) for k in ('variant','model_seed','output_root')):
                raise ValueError('Resolved prediction differs from launch matrix')
        if resolved['variant']==selection['selected_variant']:
            if (resolved.get('source_output')!=SOURCE+'/'+rid+'/source' or resolved.get('fusion_source_root')!=SOURCE or
                    resolved.get('backend_flags')!=FULL_FP32_POLICY or resolved.get('numerical_policy')!=FULL_FP32_POLICY):
                raise ValueError('Selected source path or actual prediction FP32 mismatch')
    if require_classes and data.get('classes')!=CLASSES:
        raise ValueError('Frozen class map differs')
    return rows


def collect(root, output):
    inspect(output,run=RUN,release=RELEASE)
    data=json.loads((output/'readback.json').read_text(encoding='utf-8'))
    if not data.get('pipeline') or data['pipeline'].get('status')!='ANALYZED':return False
    config=root/'experiments/cvs_response_fusion_clean/configs'
    selection=json.loads((config/'frozen_selection.json').read_text(encoding='utf-8'))
    spec=json.loads((config/'launch_spec.json').read_text(encoding='utf-8'))
    rows=validate_terminal(data,selection,spec,require_classes=False)
    source=next(r['resolved'] for r in rows if r['resolved']['variant']==selection['selected_variant'])
    # Only the already validated source physical/class contract is read here.
    script=('import json\nfrom pathlib import Path\n'
            'actual=json.loads(Path('+repr(source['source_output']+'/source_contract.json')+').read_text())\n'
            'expected=json.loads(Path('+repr(source['source_contract'])+').read_text())\n'
            'if any(actual.get(k)!=v for k,v in expected.items()):raise ValueError("Source physical contract changed")\n'
            'print(json.dumps(actual["classes"]))\n')
    data['classes']=json.loads(ssh(script))
    validate_terminal(data,selection,spec)
    marker=data.get('marker') or {}
    if any(marker.get(k)!=v for k,v in dict(status='SCORED_COMPLETE',models=11,seeds=4,rows=44,records=352,view='clean',query_count=168000).items()):
        raise ValueError('Independent scorer not complete for forty rows')
    evidence=root/'automation_reports/CV-SincNet'/RUN/'evidence';evidence.mkdir(parents=True,exist_ok=True)
    for name,value in [('final_readback.json',data),('performance_selection.json',selection),('clean_scored_results.json',data['results'])]:
        (evidence/name).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED',rows=44,source_run=SOURCE_RUN,clean_run=RUN)))
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.root,a.output)
