"""One immutable archive and one source-only two-teacher launch owner."""
import argparse
from . import design as d
from experiments.cvs_urban_bridge50 import publish as previous


def publish(output):
    before_d,before_remote=previous.d,previous.REMOTE
    try:
        previous.d=d
        previous.REMOTE=before_remote.replace('experiments/cvs_urban_bridge50','experiments/cvs_crossfit_teacher50').replace(
            'experiments.cvs_urban_bridge50.dispatch','experiments.cvs_crossfit_teacher50.dispatch')
        return previous.publish(output)
    finally:previous.d,previous.REMOTE=before_d,before_remote


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();publish(a.output)
