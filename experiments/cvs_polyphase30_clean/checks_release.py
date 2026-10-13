"""Check source import closure in the actual selected release directories."""
import json,subprocess,sys,tempfile
from pathlib import Path
from .publish import configure
from experiments.cvs_report_ssl import publish as transport
from . import design as d

def run():
 configure()
 with tempfile.TemporaryDirectory(prefix='sampling_release_import_',dir=d.ROOT/'local_artifacts') as folder:
  root=Path(folder)
  if not root.resolve().is_relative_to((d.ROOT/'local_artifacts').resolve()):raise ValueError('Release fixture must remain inside artifacts')
  # Working-tree source, so a packaging fix can be tested before its commit.
  for rel in transport.PATHS:
   source=d.ROOT/rel
   if source.is_file():files=[source]
   elif source.is_dir():files=[p for p in source.rglob('*') if p.is_file() and p.suffix in {'.py','.json','.md'}]
   else:raise FileNotFoundError(rel)
   for source in files:
    dest=root/source.relative_to(d.ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
  payload="import sys,json;sys.path[:0]=['.','code'];import experiments.cvs_polyphase30_clean.preflight as module;print(json.dumps({'status':'PASS','import_file':module.__file__}))"
  result=subprocess.run([sys.executable,'-X','utf8','-c',payload],cwd=root,text=True,capture_output=True)
  if result.returncode:raise RuntimeError(result.stderr)
  value=json.loads(result.stdout);assert Path(value['import_file']).resolve().is_relative_to(root.resolve())
 return dict(status='PASS',isolated_release_import=True,workspace_import_fallback=False,query_or_checkpoint_read=False)

if __name__=='__main__':print(json.dumps(run()))
