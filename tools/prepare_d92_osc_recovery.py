"""Preregister unchanged OSC recovery; retain both failed r01 runs verbatim."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WS = Path('E:/type10-7')


def replace(value):
    if isinstance(value, dict): return {k: replace(v) for k, v in value.items()}
    if isinstance(value, list): return [replace(v) for v in value]
    if not isinstance(value, str): return value
    for cohort in ('rx3', 'rx1'):
        value = value.replace(f'20260929-phase2-d92-osc-repeat-{cohort}-m4-r01',
                              f'20260929-phase2-d92-osc-repeat-{cohort}-m4-r02')
        value = value.replace(f'd92_osc_repeat_{cohort}_20260929_r01', f'd92_osc_repeat_{cohort}_20260929_r02')
        value = value.replace(f'configs/d92_osc_repeat_{cohort}_20260929.json',
                              f'configs/d92_osc_repeat_{cohort}_recovery_20260929.json')
        value = value.replace(f'configs/d92_osc_repeat_{cohort}_data_20260929.json',
                              f'configs/d92_osc_repeat_{cohort}_recovery_data_20260929.json')
    return value


def main():
    documents = {}
    for cohort in ('rx3', 'rx1'):
        config = f'configs/d92_osc_repeat_{cohort}_20260929.json'
        original = json.loads((ROOT/config).read_text(encoding='utf-8'))
        old = original['run_id']
        record = json.loads((WS/'automation_reports/CV-SincNet'/old/'experiment.json').read_text(encoding='utf-8'))
        if record['status'] != 'FAILED': raise ValueError('Prior run has no registered terminal failure')
        spec = replace(original)
        spec['replaces_run_id'] = old
        spec['code']['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        spec['description'] += ' Technical recovery of cache metadata binding only; fixed algorithm, data, model and matrix unchanged.'
        spec['notes'] += ['Both r01 cohorts failed during cache provenance validation before any fit or scoring; preserve all prior outputs.',
                          'Recovery corrects producer metadata binding semantics only; no formula/parameter/model/environment/data changes.']
        if spec['confirmation']['candidate'] != original['confirmation']['candidate']: raise ValueError('Formula changed')
        documents[replace(config)] = spec
        data = f'configs/d92_osc_repeat_{cohort}_data_20260929.json'
        documents[replace(data)] = json.loads((ROOT/data).read_text(encoding='utf-8'))
    if any((ROOT/path).exists() for path in documents): raise FileExistsError('Recovery already prepared')
    for path, value in documents.items():
        with (ROOT/path).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(dict(status='PREPARED_ONLY', files=list(documents), formula_changed=False)))


if __name__ == '__main__': main()
