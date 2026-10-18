"""One bounded r01 preflight interop recovery;preserve outputs and actual258NN cost."""
import os
from pathlib import Path
from . import design as d
PREVIOUS_RELEASE='cvs_FcrossRX_qualityRisk_E30_20261018_r01'
HISTORY='technical_history/r01'
FAILED_NN=258

def validate_layout(names,log_names,failure,dispatcher,prior_release,owned_dispatcher_alive=False):
 if set(names) not in ({'dispatcher.json','failure.json','logs'},{'dispatcher.json','failure.json'}) or log_names:raise ValueError('Recovery only exact failed first preflight layout,no workers/bank/smoke/row artifacts')
 if failure.get('status')!='PREFLIGHT_FAILED' or 'TypeError' not in failure.get('error','') or 'expected np.ndarray (got numpy.ndarray)' not in failure.get('error','') or not failure.get('preserve_outputs') or not failure.get('no_automatic_retry'):raise ValueError('Only confirmed r01 NumPy/Torch boundary failure is recoverable')
 if dispatcher.get('pid')!=1596070 or dispatcher.get('cwd')!=prior_release or dispatcher.get('argv')!=[prior_release+'/experiments/cvs_quality_risk30/runner.py'] or dispatcher.get('owner')!='codex/root/practical-qualityRisk' or owned_dispatcher_alive:raise ValueError('Original run-owned failed dispatcher provenance/state differs')
 return dict(status='CONFIRMED_R01_PREFLIGHT_FAILURE',failed_layer='NumPy_to_Torch_bank_IQ_bridge',actual_NN=258,real_parent_no_query_NN=2,ordinary_primary_NN=128,ordinary_second_NN=128,mixed_NN=0,raw_NN=0,optimizer_updates=0,real_IQ=0,bank_generated=False,training_workers_started=False,target_access=False)

def inspect_failure():
 prior=(Path(d.PROJECT)/'releases'/PREVIOUS_RELEASE).as_posix();failure=d.read(d.BASE/'failure.json');dispatcher=d.read(d.BASE/'dispatcher.json');pid=dispatcher.get('pid');proc=Path('/proc')/str(pid);alive=False
 if proc.exists():
  try:alive=prior+'/experiments/cvs_quality_risk30/runner.py' in (proc/'cmdline').read_bytes().decode().split(chr(0))[:-1] and os.readlink(proc/'cwd')==prior
  except FileNotFoundError:pass
 return validate_layout([p.name for p in d.BASE.iterdir()],[p.name for p in (d.BASE/'logs').iterdir()] if (d.BASE/'logs').is_dir() else [],failure,dispatcher,prior,alive)

def prepare_run():
 if d.RELEASE!='cvs_FcrossRX_qualityRisk_E30_20261018_r02':raise ValueError('Only unique r02 technical recovery')
 value=inspect_failure();history=d.BASE/HISTORY;history.mkdir(parents=True,exist_ok=False)
 for name in ('failure.json','dispatcher.json'):(d.BASE/name).rename(history/name)
 cost=dict(total_NN=258,real_parent_forward_packets=2,synthetic_ordinary_forward_packets=256,optimizer_updates=0,real_IQ=0,propagation_simulations=0)
 d.write(history/'cost.json',cost)
 if not (d.BASE/'logs').exists():(d.BASE/'logs').mkdir()
 return dict(history=HISTORY+'/cost.json',prior_failed_NN=258)
