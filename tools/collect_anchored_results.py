"""Read the completed run's full text evidence without modifying N607."""
import io,json,subprocess,tarfile
from pathlib import Path
PAYLOAD = r'''
import io,json,sys,tarfile
from pathlib import Path
b=Path('/home/szu2070436088/2510044040/CV-SincNet')
run='core90_anchored_s392005_20260911_r1'
roots=[b/'runs'/run,b/'logs'/run]
paths=[p for root in roots for p in root.rglob('*') if p.is_file()]
paths += [b/'logs'/(run+'.coordinator.log')]
inventory=[dict(path=str(p.relative_to(b)),size=p.stat().st_size) for p in paths]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
    data=json.dumps(inventory).encode(); info=tarfile.TarInfo('inventory.json');info.size=len(data);t.addfile(info,io.BytesIO(data))
    for p in paths:
        if p.suffix in {'.json','.jsonl','.csv','.log','.md'}:
            t.add(p,arcname=str(p.relative_to(b)),recursive=False)
'''
if __name__=='__main__':
    out=Path('analysis/core90_anchored_results_20260912');out.mkdir(parents=True,exist_ok=True)
    r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=PAYLOAD.encode(),capture_output=True,timeout=60)
    if r.returncode:raise RuntimeError(r.stderr.decode())
    with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as t:
        count=len(t.getmembers())
        for m in t.getmembers():
            dest=(out/m.name).resolve();assert dest.is_relative_to(out.resolve())
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(t.extractfile(m).read())
    print(json.dumps(dict(bytes=len(r.stdout),files=count)))
