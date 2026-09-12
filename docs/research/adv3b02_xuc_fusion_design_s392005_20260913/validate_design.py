"""Validate the delivered specification, arithmetic, references and mirrors."""
from __future__ import annotations
import argparse
import ast
import csv
import itertools
import json
import math
from pathlib import Path
import random
import re


def validate(root):
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in {'.md','.py','.csv','.json'})
    for path in files:
        data = path.read_bytes()
        assert not data.startswith(b'\xef\xbb\xbf'), path
        text = data.decode('utf-8')
        assert '\ufffd' not in text, path
        if path.suffix == '.json':
            json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if path.suffix == '.py':
            ast.parse(text)
    matrix=json.loads((root/'matrix15.json').read_text(encoding='utf-8'))
    runs=matrix['runs'];by_id={r['id']:r for r in runs}
    assert len(runs)==len(by_id)==matrix['experiment_count']==15
    assert set(by_id)=={f'M{i:02d}' for i in range(15)}
    assert all(r['seed']==r['data_seed']==r['evaluation_seed']==392005 for r in runs)
    assert all(r['initialization']=='scratch' and r['epochs']==200 for r in runs)
    assert all(r['checkpoint_selection']=='fixed_final_E200' for r in runs)
    assert matrix['launch'] is False
    assert [r['id'] for r in runs if r['primary']]==['M12']
    actual={(int(by_id[i]['x_enabled']),int(by_id[i]['u_enabled']),int(by_id[i]['control']=='legacy_both')) for i in matrix['factorial_rows']}
    assert actual==set(itertools.product((0,1),repeat=3))
    assert all(by_id[i]['carrier']=='v2_grid' and by_id[i]['pipeline']=='core90' for i in matrix['factorial_rows'])
    second=matrix['control_curriculum_factorial']
    assert set(second.values())=={'M05','M12','M13','M14'}
    assert {(by_id[i]['control']=='reliable_both',by_id[i]['curriculum']=='exposure_matched_capability') for i in second.values()}==set(itertools.product((False,True),repeat=2))
    assert all(by_id[i]['x_enabled'] and by_id[i]['u_enabled'] for i in second.values())
    assert all(by_id[i]['source_audit_policy']=='cstar_isolated_passive_or_active' for i in second.values())
    assert all(r['cstar_action_enabled']==(r['control']=='reliable_both') for r in runs)
    assert all(r['ticket_curriculum_enabled']==(r['curriculum']=='exposure_matched_capability') for r in runs)
    assert set(matrix['cstar']['window_boundaries_split_at_epochs']) >= {8,12,20,25,41,45,80,91,131}
    assert set(matrix['cstar']['ticket_schedule']['fields']) >= {'origin_epoch','origin_batch_index','origin_main_step','objective_stage_weights'}
    assert 'observation_id' in matrix['cstar']['observation_clock']['unit']
    native=json.loads((root/'a1_native_recipe_reference.json').read_text(encoding='utf-8'))
    for rid,weight in [('M09','0'),('M10','0.05')]:
        assert by_id[rid]['recipe_reference']=='a1_native_recipe_reference.json'
        options=native['rows'][rid]['reference_options']
        assert options['--use_muse_ssdg']=='true' and options['--a1_runtime_fast']=='true'
        assert options['--a1_ecrs_cross_rx_scope']=='clean' and options['--a1_ecrs_cross_rx_weight']==weight
        assert options['--amp']=='false' and options['--from_scratch']=='true'
        assert not options['--baseline_ckpt'] and not options['--teacher_ckpt']
    assert all(sum(v.values())==0 and set(v)<=set(by_id) for v in matrix['interaction_contrasts'].values())
    assert matrix['expected_all_predictions']==15*168000*4==10080000
    assert matrix['nominal_main_update_opportunities']==sum(r['epochs']*r['steps_per_epoch'] for r in runs)==216200
    with (root/'matrix15.csv').open(encoding='utf-8',newline='') as stream:
        csvrows=list(csv.DictReader(stream))
    assert [r['id'] for r in csvrows]==[r['id'] for r in runs]
    assert all(int(r['seed'])==392005 for r in csvrows)
    evidence=json.loads((root/'historical_evidence.json').read_text(encoding='utf-8'))
    assert len(evidence['history'])==16 and len(evidence['contrasts'])==6
    for row in evidence['history']:
        if 'counts' in row:
            for scene,c in row['counts'].items():
                assert c['total']==row['n_per_scene'] and 0<=c['correct']<=c['total']
                assert math.isclose(100*c['correct']/c['total'],row['scenes'][scene],abs_tol=1e-10)
        if 'scenes' in row:
            mean=sum(v for s,v in row['scenes'].items() if s!='clean')/3
            assert math.isclose(mean,row['leo_mean'],abs_tol=1e-10)
    # Verify the report's equal-grid variance decomposition without importing a training runtime.
    rng=random.Random(392005);p,q,d=4,4,7
    g=[[[rng.uniform(-1,1) for _ in range(d)] for _ in range(q)] for _ in range(p)]
    mu=[sum(g[t][r][j] for t in range(p) for r in range(q))/(p*q) for j in range(d)]
    a=[[sum(g[t][r][j] for r in range(q))/q-mu[j] for j in range(d)] for t in range(p)]
    b=[[sum(g[t][r][j] for t in range(p))/p-mu[j] for j in range(d)] for r in range(q)]
    e=[[[g[t][r][j]-mu[j]-a[t][j]-b[r][j] for j in range(d)] for r in range(q)] for t in range(p)]
    lhs=sum((g[t][r][j]-mu[j]-a[t][j])**2 for t in range(p) for r in range(q) for j in range(d))/(p*q)
    rhs=sum(v*v for x in b for v in x)/q+sum(v*v for x in e for y in x for v in y)/(p*q)
    assert math.isclose(lhs,rhs,rel_tol=1e-12,abs_tol=1e-12)
    report=(root/'report.md').read_text(encoding='utf-8')
    for rid in by_id:
        assert f'|{rid}|' in report or f'|**{rid}**|' in report,rid
    checked=0
    for dest in re.findall(r'\]\(([^)]+)\)',report):
        if dest.startswith(('codex://','https://','http://','#')):continue
        path=Path(dest)
        if not path.is_absolute():path=root/path
        assert path.exists(),f'Broken local link: {dest}'
        checked+=1
    return dict(status='VERIFIED',scope='design_files_and_existing_aggregate_arithmetic_only',
        files=len(files),matrix_rows=len(runs),factorial_cells=len(actual),secondary_factorial_cells=4,
        history_rows=16,local_links_checked=checked,encoding='UTF-8 no BOM',new_training_tests_run=False,
        remote_training_started=False,variance_identity_error=abs(lhs-rhs))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--mirror',type=Path);parser.add_argument('--write-result',action='store_true')
    args=parser.parse_args();result=validate(args.directory)
    if args.mirror:
        left={p.name:p.read_bytes() for p in args.directory.iterdir() if p.is_file() and p.name not in {'validation.json','delivery.md'}}
        right={p.name:p.read_bytes() for p in args.mirror.iterdir() if p.is_file() and p.name not in {'validation.json','delivery.md'}}
        assert left==right,'mirror differs';result['mirror']='VERIFIED byte_equal'
    if args.write_result:
        (args.directory/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False))
