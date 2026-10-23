"""Replay final external/fairness CPU analyses from portable frozen metadata."""
import argparse
import csv
import json
import math
from pathlib import Path


def compare_csv(one, two):
    with Path(one).open(encoding='utf-8-sig', newline='') as f:
        a = csv.reader(f)
        with Path(two).open(encoding='utf-8-sig', newline='') as g:
            b = csv.reader(g)
            if next(a) != next(b):
                raise ValueError('Reproduction CSV header differs')
            numeric = rows = 0
            maximum = 0.
            while True:
                x, y = next(a, None), next(b, None)
                if x is None or y is None:
                    if x != y:
                        raise ValueError('Reproduction CSV length differs')
                    break
                if len(x) != len(y):
                    raise ValueError('Reproduction CSV width differs')
                rows += 1
                for u, v in zip(x, y):
                    try:
                        p, q = float(u), float(v)
                    except ValueError:
                        if u != v:
                            raise ValueError('Reproduction text field differs')
                        continue
                    if not math.isfinite(p) or not math.isfinite(q) or abs(p - q) > 1e-10:
                        raise ValueError('Reproduction numeric field differs')
                    numeric += 1
                    maximum = max(maximum, abs(p - q))
    return dict(rows=rows, numeric_cells=numeric, maximum_absolute_difference=maximum)


class PortableInput:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()

    def expand(self, value):
        if isinstance(value, dict):
            return {k: self.expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.expand(v) for v in value]
        if isinstance(value, str) and (value == 'PROJECT' or value.startswith('PROJECT/')):
            return (self.folder / 'remote').as_posix() + value[len('PROJECT'):]
        return value

    def read(self, path):
        path = Path(self.expand(str(path)))
        if not path.is_absolute():
            path = self.folder / path
        return self.expand(json.loads(path.read_text(encoding='utf-8-sig')))


def run(folder, output):
    from . import analyze_external_dura, analyze_gfactor
    source = PortableInput(folder)
    manifest = source.read('manifest.json')
    if (manifest['status'] != 'ALL_EFFECTIVE_PHASES_COMPLETE' or manifest['queued'] or manifest['running']
            or manifest['historical_failed'] != 1 or not manifest['scientific_jobs_unchanged_across_handoff']):
        raise ValueError('Final effective phase closure is not established')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name, module in (('external', analyze_external_dura), ('gfactor', analyze_gfactor)):
        c = source.read(manifest['analysis_configs'][name])
        original = Path(c['output_root'])
        c['output_root'] = str(out / name)
        saved = module.read
        module.read = source.read
        try:
            module.run(c)
        finally:
            module.read = saved
        files = {}
        for path in sorted(original.glob('*.csv')):
            files[path.name] = compare_csv(path, out / name / path.name)
        if not files:
            raise ValueError('No independent remote CSVs found')
        reports[name] = dict(status='REPRODUCED', files=files,
            numeric_cells=sum(v['numeric_cells'] for v in files.values()),
            maximum_absolute_difference=max(v['maximum_absolute_difference'] for v in files.values()))
    (out / 'reproduction_verification.json').write_text(json.dumps(reports, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(reports))
    return reports


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.input, a.output)
