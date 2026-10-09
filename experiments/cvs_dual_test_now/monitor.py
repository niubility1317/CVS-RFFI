"""Reuse verified result recount, resolving source logs from their original run."""
import io,json,tarfile
from pathlib import Path
from types import SimpleNamespace
from experiments.cvs_dual_evidence import monitor as original
from experiments.cvs_dual_test_now import driver as d
from experiments.cvs_phase1_stack.publish import ssh

def install():
    values={k:v for k,v in vars(d.source).items() if not k.startswith('__')}
    values.update(RUN=d.RUN,RELEASE=d.RELEASE,ROOT=d.ROOT,BASE=d.BASE,OWNER='codex/root/dual-evidence-immediate-test-20261009')
    original.d=SimpleNamespace(**values)
    original.ARTIFACT=d.ROOT/'local_artifacts'/d.RELEASE
    original.REPORT=d.ROOT/'automation_reports/CV-SincNet'/d.RUN
    original_pull=original.pull
    def pull():
        folder=original_pull()
        names=[r['row_id']+'/source/'+n for r in d.source.rows() for n in ('completion.json','initialization.json','resolved_config.json','resolved_native_args.json','epoch_metrics.jsonl')]
        code="from pathlib import Path\nimport sys,tarfile\np=Path("+repr(d.source.BASE.as_posix())+")\nnames="+repr(names)+"\nwith tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:\n for n in names:\n  q=p/n\n  if not q.is_file():raise FileNotFoundError(q)\n  t.add(q,arcname=n,recursive=False)\n"
        blob=ssh(code)
        with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
            for m in archive.getmembers():
                if not m.isfile() or not (folder/m.name).resolve().is_relative_to(folder.resolve()):raise ValueError('Unsafe source artifact')
            archive.extractall(folder)
        return folder
    original.pull=pull
    return original

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    m=install()
    if a.watch:m.watch()
    else:
        s=m.readback();print(json.dumps(dict(controller=s['controller'],queue=s['queue'],failure=s['failure'],completion=s['completion'],rows=[dict(row_id=r['row_id'],pid=r['pid'],gpu=r['gpu'],actual=r['actual'],completion=r['completion']) for r in s['rows']])))
