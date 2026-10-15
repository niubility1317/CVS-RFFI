"""Use the existing cross-owner lock and inspect actual/pre-CUDA processes."""
import os,subprocess
from pathlib import Path
from contextlib import contextmanager
from . import design as d
LIMIT=3;MIN_FREE_MIB=12000
@contextmanager
def capacity_lock():
 import fcntl
 with (Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX);yield
def process(pid):
 p=Path('/proc')/str(pid)
 try:
  stat=(p/'stat').read_text().rsplit(')',1)[1].split()
  if stat[0] in ('Z','X','x'):return None
  try:env=dict(v.split('=',1) for v in (p/'environ').read_bytes().decode().split('\0') if '=' in v)
  except PermissionError:env={}
  return dict(pid=int(pid),start_ticks=int(stat[19]),cwd=str((p/'cwd').resolve()),argv=[s for s in (p/'cmdline').read_bytes().decode().split('\0') if s],gpu=env.get('CUDA_VISIBLE_DEVICES'))
 except (FileNotFoundError,ProcessLookupError,PermissionError,UnicodeError):return None
def capacity():
 result={};uuids={}
 for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines():
  index,uuid,free=[x.strip() for x in line.split(',')];index=int(index);uuids[uuid]=index;result[index]=dict(pids=set(),free_MiB=int(free),uuid=uuid)
 for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines():
  if not line.strip():continue
  pid,uuid=[x.strip() for x in line.split(',')]
  if uuid in uuids:result[uuids[uuid]]['pids'].add(int(pid))
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():continue
  p=process(int(path.name))
  if not p:continue
  # Empty CUDA means CPU. Account numeric, UUID and multi-device reservations.
  for token in str(p['gpu'] or '').split(','):
   gpu=int(token) if token.isdigit() else uuids.get(token)
   if gpu in result:result[gpu]['pids'].add(p['pid'])
 return result
def serial(caps):return {str(g):dict(v,pids=sorted(v['pids'])) for g,v in caps.items()}
def choose(caps):
 choices=[g for g,v in caps.items() if len(v['pids'])<LIMIT and v['free_MiB']>=MIN_FREE_MIB]
 return min(choices,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_MiB'],g)) if choices else None
