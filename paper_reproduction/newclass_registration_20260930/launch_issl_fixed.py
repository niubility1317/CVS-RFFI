"""Exclusive native launch for a preregistered ISSL repair diagnostic."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from issl_fixed import validate_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_config(config)
    output = Path(config['output']).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix('.stdout.log')
    launch_path = output.with_suffix('.launch.json')
    if any(path.exists() for path in (output, log_path, launch_path)):
        raise FileExistsError('Run output or launch evidence already exists')
    command = [sys.executable, '-X', 'utf8', str(Path(__file__).with_name('issl_fixed.py').resolve()), '--config', str(config_path)]
    with log_path.open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=Path.cwd())
        with launch_path.open('x', encoding='utf-8') as evidence:
            json.dump(dict(pid=process.pid, command=command, cwd=str(Path.cwd()), log=str(log_path), config=str(config_path)), evidence, indent=2)
            evidence.write('\n')
        print(json.dumps(dict(pid=process.pid, log=str(log_path), output=str(output))), flush=True)
        result = process.wait()
    if result: raise RuntimeError(f'Execution failed ({result}); preserved log: {log_path}')
    resolved = json.loads((output/'resolved_config.json').read_text(encoding='utf-8'))
    acceptance = json.loads((output/'acceptance.json').read_text(encoding='utf-8'))
    if resolved['pid'] != process.pid or acceptance['status'] != 'VERIFIED' or not (output/'predictions.npz').is_file():
        raise RuntimeError('Post-state evidence mismatch')
    print(json.dumps(dict(status='VERIFIED', run_id=config['run_id'], pid=process.pid, steps=acceptance['steps'], output=str(output))), flush=True)


if __name__ == '__main__': main()
