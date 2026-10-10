"""Immutable new release, keeping existing healthy runs untouched."""
import argparse
from pathlib import Path
from experiments.cvs_multi_disentangle import publish as transport
from experiments.cvs_multi_state_action.publish import REMOTE as PARENT_REMOTE
from . import design as d
start=PARENT_REMOTE.index('\ncheck=')+1;end=PARENT_REMOTE.index("with (release/'preflight.log')",start)
REMOTE=PARENT_REMOTE[:start]+'''check="from experiments.cvs_feature_disentangle.checks import preflight; preflight('preflight-synthetic')"
'''+PARENT_REMOTE[end:]
REMOTE=REMOTE.replace('experiments.cvs_multi_state_action.dispatch','experiments.cvs_feature_disentangle.dispatch')

def publish(output):
 transport.RUN=d.RUN;transport.RELEASE=d.RELEASE;transport.REMOTE=REMOTE
 transport.PATHS=[*transport.PATHS,'experiments/cvs_multi_action_audit','experiments/cvs_multi_state_action',
   'experiments/cvs_legacy_no_sat','experiments/cvs_multi_action_risk','experiments/cvs_feature_disentangle',
   'code/sat_channel.py','code/training_controls.py']
 transport.publish(output)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);publish(p.parse_args().output)
