"""Read-only checkout, syntax and original CLI audit; does not train models."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('E:/type10-7/external_sources/rffi_labeled_da_20260915')

def git(root, *args):
    p = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
    return dict(returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())

def main():
    results = {}
    for name, entries in {
        'RadioFingerprinting': ['fine-tuning/main.py', 'fine-tuning/finetune.py'],
        'RadioNet': ['model_training.py', 'finetune.py', 'model_test.py'],
    }.items():
        root = ROOT / name
        tracked = git(root, 'ls-files')['stdout'].splitlines()
        syntax = []
        for rel in tracked:
            if rel.endswith('.py'):
                try:
                    compile((root/rel).read_bytes(), str(root/rel), 'exec')
                except (SyntaxError, ValueError) as e:
                    syntax.append(dict(path=rel, error=str(e)))
        probes = []
        for rel in entries:
            env = dict(os.environ, KERAS_BACKEND='torch')
            try:
                p = subprocess.run([sys.executable, str(root/rel), '--help'],
                                   cwd=(root/rel).parent, env=env, capture_output=True,
                                   text=True, encoding='utf-8', errors='replace', timeout=45)
                probes.append(dict(entry=rel, returncode=p.returncode,
                                   stdout=p.stdout[-1500:], stderr=p.stderr[-3000:]))
            except subprocess.TimeoutExpired:
                probes.append(dict(entry=rel, status='TIMEOUT_45S'))
        results[name] = dict(commit=git(root, 'rev-parse', 'HEAD')['stdout'],
                             fsck=git(root, 'fsck', '--full'), status=git(root, 'status', '--porcelain'),
                             tracked_files=len(tracked), missing_tracked=[p for p in tracked if not (root/p).exists()],
                             python_files=sum(p.endswith('.py') for p in tracked), syntax_errors=syntax,
                             entrypoint_probes=probes)
    results['environment'] = dict(python=sys.executable, dependencies={
        m: bool(importlib.util.find_spec(m)) for m in ['tensorflow','keras','numpy','scipy','sklearn','h5py']})
    out=ROOT/'author_code_audit_20260915.json'
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    for name in ['RadioFingerprinting','RadioNet']:
        r=results[name]
        print(name, r['tracked_files'], r['python_files'], 'missing',r['missing_tracked'], 'syntax',r['syntax_errors'])
        for p in r['entrypoint_probes']:
            print(p['entry'],p.get('returncode'),p.get('stderr','').splitlines()[-1:])

if __name__=='__main__':
    main()
