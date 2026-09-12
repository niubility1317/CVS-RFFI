"""Validate acceptance evidence and mirror only these two owned report folders."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path('E:/type10-7')
OUT = Path(__file__).resolve().parent
DESIGN = OUT.parent / 'adv3b02_xuc_fusion_design_s392005_20260913'
WORKTREE = ROOT / 'github_publish/CVS-RFFI-repo/.worktrees/adv3b02-xuc-fusion-design-20260913'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def validate():
    probes = read(OUT / 'probe_results.json')
    assert probes['status'] == 'VERIFIED' and len(probes['jobs']) == 7
    assert all(j['returncode'] == 0 for j in probes['jobs'])
    suites = {}
    for path in OUT.glob('*.xml'):
        rows = [dict(s.attrib) for s in ET.parse(path).iter('testsuite')]
        assert rows and all(int(s['failures']) == int(s['errors']) == int(s['skipped']) == 0 for s in rows)
        suites[path.name] = sum(int(s['tests']) for s in rows)
    assert suites == {'x_unit_tests.xml':20, 'u_normalized_tests.xml':5, 'u_roles_tests.xml':8, 'game_solver_tests.xml':12}
    real = read(OUT / 'kernel_probe.json')
    assert real['model_kwargs']['num_domains'] == 15 and real['identity_shape'] == [32, 160]
    assert real['batchnorm_modules'] == [] and len(real['mixstyle_modules']) == 2
    assert real['device'] == 'cpu' and real['seed'] == 392005
    for key in ['x', 'u']:
        assert real[key]['weighted_feature_gradient_norm'] > 0
        assert real[key]['identity_active'] and not real[key]['domain_active'] and not real[key]['adversary_active']
    assert not real['full_core90_objective_tested'] and not real['cstar_tested'] and not real['extragradient_transaction_tested']
    runtime = read(OUT / 'runtime_probe.json')
    assert runtime['interaction_mode'] == 'normalized' and runtime['lambda_cross'] == .01
    assert runtime['successful_steps'] == 1 and not runtime['response_enabled'] and not runtime['decision_enabled']
    # Read-only inventory of the original delivery and current candidate code.
    original = subprocess.run(['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', '273f4edfa393eedabc1dc9fe18cdbf6b1bac11dd'],
                              cwd=WORKTREE, capture_output=True, text=True, encoding='utf-8', check=True).stdout.splitlines()
    assert len(original) == 10 and all(p.startswith('docs/research/adv3b02_xuc_fusion_design_s392005_20260913/') for p in original)
    inventory = []
    pattern = 'class XUC|def train_core90_xuc|xuc_fusion|reliable_both|exposure_matched_capability'
    for code in [ROOT / 'code', WORKTREE / 'code']:
        scan = subprocess.run(['rg', '-l', pattern, str(code), '-g', '*.py'], capture_output=True, text=True, encoding='utf-8')
        assert scan.returncode == 1 and not scan.stderr, (scan.returncode, scan.stdout, scan.stderr)
        inventory.append(dict(root=str(code), pattern=pattern, matching_python_files=[]))
    matrix = read(DESIGN / 'matrix15.json')
    assert matrix['status'] == 'DESIGN_COMPLETE_NOT_IMPLEMENTED_NOT_LAUNCHED' and not matrix['launch']
    summary = dict(verdict='NOT_ACCEPTED_IMPLEMENTATION_ABSENT', review_complete=True,
        existing_component_tests='VERIFIED', pytest_passed=sum(suites.values()), pytest_suites=suites,
        synthetic_runtime_and_model_probes='VERIFIED_WITH_EXPLICIT_LIMITED_SCOPE',
        design_revision='VERIFIED_SPEC_ONLY', fusion_implementation='NOT_IMPLEMENTED',
        fusion_runtime_activation='NOT_EVALUATED_NO_RUN_ARTIFACT', formal_scoring='NOT_EXECUTED',
        source_delivery_commit='273f4edfa393eedabc1dc9fe18cdbf6b1bac11dd', original_delivery_files=original,
        current_local_inventory=inventory, experiment_count=15, seed=392005,
        fixes={'ticket_schedule':'SPEC_UPDATED_IMPLEMENTATION_PENDING',
               'native_bridge_recipe':'REFERENCE_ADDED_ADAPTER_PENDING',
               'observation_clock':'SPEC_UPDATED_IMPLEMENTATION_PENDING',
               'passive_audit_policy':'EXPLICIT_CONFIG_ADDED_IMPLEMENTATION_PENDING',
               'batchnorm_activation_claim':'CORRECTED_TO_NOT_APPLICABLE'},
        remote_training_started=False, checkpoint_loaded=False, scientific_performance_claim='NOT_EVALUATED')
    write(OUT / 'acceptance_summary.json', summary)
    counts = {'files':0, 'local_links':0}
    for folder in [OUT, DESIGN]:
        for path in folder.iterdir():
            if not path.is_file() or path.suffix not in {'.md','.py','.json','.csv','.log','.xml'}:
                continue
            raw = path.read_bytes()
            assert not raw.startswith(b'\xef\xbb\xbf'), path
            source = raw.decode('utf-8')
            assert '\ufffd' not in source, path
            if path.suffix == '.json':
                json.loads(source, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
            elif path.suffix == '.py':
                ast.parse(source)
            elif path.suffix == '.md':
                for ref in re.findall(r'\]\(([^)]+)\)', source):
                    if ref.startswith(('https://','http://','codex://','#')):
                        continue
                    target = Path(ref)
                    if not target.is_absolute():
                        target = folder / target
                    assert target.exists(), (path, ref)
                    counts['local_links'] += 1
            counts['files'] += 1
    return dict(status='VERIFIED', scope='acceptance_evidence_and_report_consistency_only', **counts,
                pytest_passed=sum(suites.values()), implementation_verdict=summary['verdict'])


def mirror():
    base = (WORKTREE / 'docs/research').resolve()
    result = {}
    for folder in [OUT, DESIGN]:
        target = (base / folder.name).resolve()
        assert target.parent == base
        target.mkdir(parents=True, exist_ok=True)
        names = []
        for path in folder.iterdir():
            if path.is_file() and path.suffix in {'.md','.py','.json','.csv','.log','.xml'}:
                shutil.copy2(path, target / path.name)
                assert path.read_bytes() == (target / path.name).read_bytes()
                names.append(path.name)
        result[folder.name] = dict(status='VERIFIED_BYTE_EQUAL', files=len(names))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mirror', action='store_true')
    args = parser.parse_args()
    result = validate()
    if args.mirror:
        result['mirror'] = mirror()
    print(json.dumps(result, ensure_ascii=False))
