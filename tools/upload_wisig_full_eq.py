"""Copy validated ZIPs over the existing N607 route, preserving existing data."""
import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import time
import zipfile


def run(args):
    source = Path(args.source).resolve()
    validation = json.loads(Path(args.validation).read_text(encoding='utf-8'))
    files = sorted(source.glob('*.zip'))
    if validation['status'] != 'VERIFIED' or len(files) != validation['archives']:
        raise ValueError('Require the completed local data verification')
    if sum(p.stat().st_size for p in files) != validation['compressed_bytes']:
        raise ValueError('Input size changed since validation')
    dest = PurePosixPath(args.destination)
    if not dest.is_absolute() or '..' in dest.parts or len(dest.parts) < 6:
        raise ValueError('Expected an explicit ordinary-account dataset directory')
    connection = ['-F', str(Path(args.ssh_config).resolve()), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    records = []

    def remote(code):
        result = subprocess.run(['ssh', *connection, '-T', 'N607', 'python3 -'],
                                input=code.encode('utf-8'), capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors='replace') + result.stdout.decode(errors='replace'))
        return json.loads(result.stdout)

    initial = remote("import json,os,shutil\nfrom pathlib import Path\np=Path(" + repr(str(dest)) + ")\n"
                     "assert p.parts[2]==os.getlogin() if False else p.parts[2]==__import__('getpass').getuser()\n"
                     "assert shutil.disk_usage(p.parent if p.parent.exists() else Path.home()).free>" + str(validation['compressed_bytes'] * 2) + "\n"
                     "p.mkdir(parents=True,exist_ok=True)\nprint(json.dumps({'directory':str(p),'files':{x.name:x.stat().st_size for x in p.iterdir() if x.is_file()}}))")
    (output / 'initial.json').write_text(json.dumps(initial, indent=2), encoding='utf-8')
    for path in files:
        target = str(dest / path.name)
        partial = target + '.uploading'
        old_size = initial['files'].get(path.name)
        if old_size is not None and old_size != path.stat().st_size:
            raise FileExistsError('Existing final file differs: ' + path.name)
        if old_size is None:
            if path.name + '.uploading' in initial['files']:
                raise FileExistsError('Reconcile prior partial upload before retry: ' + path.name)
            print(json.dumps({'event':'COPY_START','file':path.name,'bytes':path.stat().st_size}), flush=True)
            subprocess.run(['scp', *connection, str(path), 'N607:' + partial], check=True)
        with zipfile.ZipFile(path) as z:
            expected = [(i.filename, i.file_size, i.CRC) for i in z.infolist()]
        evidence = remote("import json,zipfile\nfrom pathlib import Path\np=Path(" + repr(target if old_size is not None else partial) + ")\n"
                          "assert p.stat().st_size==" + str(path.stat().st_size) + "\n"
                          "with zipfile.ZipFile(p) as z:\n assert [(i.filename,i.file_size,i.CRC) for i in z.infolist()]==" + repr(expected) + "\n assert z.testzip() is None\n"
                          + ("final=Path(" + repr(target) + ")\nassert not final.exists()\np.rename(final)\n" if old_size is None else '')
                          + "print(json.dumps({'file':" + repr(path.name) + ",'bytes':p.stat().st_size if p.exists() else Path(" + repr(target) + ").stat().st_size,'CRC':'PASS'}))")
        records.append(evidence)
        state = dict(status='UPLOADING' if len(records) < len(files) else 'VERIFIED',
                     completed=len(records), total=len(files), completed_bytes=sum(r['bytes'] for r in records),
                     elapsed_seconds=time.time()-started, destination=str(dest), files=records)
        (output / 'progress.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        print(json.dumps(state | {'files':None}), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','validation','destination','ssh-config','output'):
        p.add_argument('--'+name,required=True)
    run(p.parse_args())
