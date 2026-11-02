"""Wait for a single exact direct-C run and then launch its independent readout."""
import argparse
from pathlib import Path
import time
from .io import read, require
from .r42_direct_c_analysis import analyze


def is_controller(argv, runtime):
    modules = {'experiments.pamf_phase2_practical.r42_direct_c_dispatch',
               'experiments.pamf_phase2_practical.r42_direct_c_takeover'}
    if argv.count('-m') != 1 or argv.count('--runtime') != 1:
        return False
    i, j = argv.index('-m'), argv.index('--runtime')
    return i + 1 < len(argv) and j + 1 < len(argv) and argv[i+1] in modules and argv[j+1] == runtime


def controller_running(proc, complete, runtime):
    try:
        command = proc.read_bytes().decode('utf-8').rstrip('\0').split('\0')
    except (FileNotFoundError, ProcessLookupError):
        if complete.exists():
            return False
        raise ValueError('Controller exited before scoring closed')
    if not is_controller(command, runtime):
        if complete.exists():
            return False
        raise ValueError('Controller identity changed before scoring closed')
    return True


def wait(runtime, native_index, output, controller_pid):
    rt = read(runtime)
    root = Path(rt['run_root'])
    complete = root / 'scoring_complete.json'
    while not complete.exists():
        require(not (root / 'failed.json').exists(), 'Direct-C worker failed; readout stopped')
        proc = Path('/proc') / str(controller_pid) / 'cmdline'
        if not controller_running(proc, complete, runtime):
            break
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
