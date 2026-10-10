"""Publish a new immutable evaluation release from verified local Git."""
import argparse
from pathlib import Path
from experiments.cvs_multi_disentangle import publish as transport
from experiments.cvs_multi_state_action.publish import REMOTE as PARENT_REMOTE
from experiments.cvs_state_test_now import design as d

REMOTE=PARENT_REMOTE
start=REMOTE.index('\ncheck=')+1
end=REMOTE.index("with (release/'preflight.log')",start)
REMOTE=REMOTE[:start]+'''check="from experiments.cvs_state_test_now.dispatch import prepare; prepare()"
'''+REMOTE[end:]
REMOTE=REMOTE.replace('experiments.cvs_multi_state_action.dispatch','experiments.cvs_state_test_now.dispatch')

def publish(output):
    transport.RUN=d.RUN;transport.RELEASE=d.RELEASE;transport.REMOTE=REMOTE
    transport.PATHS=[*transport.PATHS,'experiments/cvs_multi_action_audit','experiments/cvs_multi_state_action',
        'experiments/cvs_state_test_now','code/sat_channel.py','code/training_controls.py']
    transport.publish(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    publish(p.parse_args().output)
