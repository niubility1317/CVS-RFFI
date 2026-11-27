"""Extract an isolated pinned NumPy wheel; never mutate the shared Python env."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

p=argparse.ArgumentParser()
p.add_argument('--wheel',type=Path,required=True)
p.add_argument('--sha256',required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
assert hashlib.sha256(args.wheel.read_bytes()).hexdigest()==args.sha256
if args.output.exists(): raise FileExistsError(args.output)
with zipfile.ZipFile(args.wheel) as archive:
 for name in archive.namelist():
  path=Path(name)
  if path.is_absolute() or '..' in path.parts: raise ValueError(name)
 args.output.mkdir(parents=True)
 archive.extractall(args.output)
(args.output/'runtime_source.json').write_text(json.dumps({'numpy':'1.26.4','wheel':args.wheel.name,'sha256':args.sha256,
  'shared_environment_modified':False},indent=2)+'\n',encoding='utf-8')
print('ISOLATED_RUNTIME_PREPARED',args.output)
