"""Create non-destructive author checkouts with API-only compatibility patches."""
import argparse
import ast
import difflib
import json
from pathlib import Path
import re
import shutil
import subprocess


def prepare(sources, output):
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, commit in [('RadioFingerprinting', '23d1fd4aef2f1dd254c5596680ae1da66f416916'),
                         ('RadioNet', '64f4b0a3dfb48b4a389e4d177e9ffa35010e17f5')]:
        src, dst = sources / name, output / name
        actual = subprocess.check_output(['git', '-C', str(src), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != commit:
            raise ValueError(f'{name}: unexpected source commit {actual}')
        if subprocess.check_output(['git', '-C', str(src), 'diff', 'HEAD', '--'], text=True):
            raise ValueError(f'{name}: tracked source modifications')
        files = subprocess.check_output(['git', '-C', str(src), 'ls-files'], text=True).splitlines()
        for rel in files:
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, target)
        changed = []
        scope = dst / 'fine-tuning' if name == 'RadioFingerprinting' else dst
        paths = list(scope.glob('*.py'))
        if name == 'RadioNet':
            paths.append(dst / 'models/rf_models.py')
        for path in paths:
            before = path.read_text(encoding='utf-8-sig')
            after = before.replace('tensorflow.keras', 'keras')
            after = after.replace('from keras.utils import np_utils', 'from keras import utils as np_utils')
            after = re.sub(r'np\.(int|float|complex)\b', r'\1', after)
            after = after.replace("'val_acc'", "'val_accuracy'")
            after = after.replace("optimizer='Adam'", "optimizer='adam'")
            if name == 'RadioNet' and path.name == 'rf_models.py':
                # Only remove eager imports unrelated to the supervised DF route.
                after = after.replace('from complexnn import ComplexConv1D, Modrelu, SpectralPooling1D, ComplexDense', '')
                after = after.replace('from complexnn import utils', '')
                after = after.replace('import af_model, af_classifier', '')
            if name == 'RadioNet' and path.name in ('finetune.py', 'model_training.py'):
                after = after.replace('import augData\n', '').replace('import get_simu_data\n', '')
            if path.name == 'finetune.py':
                # Recompile AFTER the author's unchanged weight-copy/freeze loop.
                needle = '            l1.trainable = False\n'
                assert after.count(needle) == 1
                after = after.replace(needle, needle + "\n        new_model.compile(loss='categorical_crossentropy', optimizer='adam',\n                          metrics=['accuracy', 'top_k_categorical_accuracy'])\n")
            if after != before:
                path.write_text(after, encoding='utf-8', newline='\n')
                rel = str(path.relative_to(dst)).replace('\\', '/')
                changed.append(rel)
                with (output / 'compat.patch').open('a', encoding='utf-8') as f:
                    f.writelines(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                    fromfile=f'{name}/{rel}', tofile=f'{name}/{rel}'))
        model = 'fine-tuning/radioConv.py' if name == 'RadioFingerprinting' else 'models/rf_models.py'
        def functions(p):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(p.read_text(encoding='utf-8-sig')).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert functions(src / model) == functions(dst / model), 'Core model changed'
        records[name] = dict(commit=commit, changed=changed, all_model_functions_unchanged=True)
    (output / 'manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.sources, args.output), indent=2))
