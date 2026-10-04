"""New-process execution adapter; scientific source code stays immutable."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys


def run(worker_root,config):
    release=Path(__file__).resolve().parent
    sys.path[:0]=[str(release),str(worker_root),str(worker_root/'code')]
    from fast_execution import POLICY,batch_clean,configure,incremental_native_writer
    from experiments.cvs_phase1_stack import source as s
    WORKER_COMMIT='0c73c904c8331f26254761042516f1ae43ed9c98'
    if s.ROOT!=worker_root or (worker_root/'release_commit.txt').read_text().strip()!=WORKER_COMMIT:
        raise ValueError('Original worker release mismatch')
    c=s.read(config)
    if c['stage'] not in ['r3','r4','r5','r6']:
        raise ValueError('Acceleration is only authorized for future R3–R6 rows')
    execution=dict(release=str(release),commit=(release/'release_commit.txt').read_text().strip(),policy=POLICY,
                   original_worker_commit=WORKER_COMMIT,original_worker_root=str(worker_root))
    old_make=s.make_args;old_installed=s.installed;old_write=s.write
    @contextmanager
    def installed(value):
        with old_installed(value) as native,incremental_native_writer(native):yield native
    def write(path,value):
        if Path(path).name in ['resolved_config.json','completion.json']:
            value=dict(value,execution=execution)
        return old_write(path,value)
    s.make_args=lambda *a,**kw:configure(old_make(*a,**kw))
    s.clean=batch_clean;s.installed=installed;s.write=write
    print('EXECUTION_POLICY '+s.json.dumps(execution),flush=True)
    s.train(c)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-root',type=Path,required=True);p.add_argument('--config',required=True)
    a=p.parse_args();run(a.worker_root,a.config)
