"""Wait for a single exact direct-C run and then launch its independent readout."""
import argparse
from pathlib import Path
import time
from .io import read, require
from .r42_direct_c_analysis import analyze


def wait(runtime, native_index, output, controller_pid):
    rt = read(runtime)
    root = Path(rt['run_root'])
    complete = root / 'scoring_complete.json'
    while not complete.exists():
        require(not (root / 'failed.json').exists(), 'Direct-C worker failed; readout stopped')
        proc = Path('/proc') / str(controller_pid) / 'cmdline'
        require(proc.exists(), 'Controller exited before scoring closed')
        command = proc.read_bytes().replace(b'\0', b' ').decode('utf-8')
        require('r42_direct_c_dispatch' in command and runtime in command, 'Controller identity changed')
        time.sleep(30)
    analyze(rt, native_index, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--native-index', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--controller-pid', type=int, required=True)
    args = parser.parse_args()
    wait(args.runtime, args.native_index, args.output, args.controller_pid)


if __name__ == '__main__':
    main()
