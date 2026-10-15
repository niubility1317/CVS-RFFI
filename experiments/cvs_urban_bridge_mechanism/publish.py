"""Reuse one-archive source transport with a fresh immutable registered run."""
import argparse
from . import design as d
from experiments.cvs_urban_semantic_mechanism import publish as previous


def publish(output):
    # Actual checkpoint eligibility is checked remotely before source IQ access.
    before_d,before_remote=previous.d,previous.REMOTE
    try:
        previous.d=d
        previous.REMOTE=before_remote.replace('experiments/cvs_urban_semantic_mechanism','experiments/cvs_urban_bridge_mechanism').replace(
            'experiments.cvs_urban_semantic_mechanism.probe','experiments.cvs_urban_bridge_mechanism.probe')
        return previous.publish(output)
    finally:previous.d,previous.REMOTE=before_d,before_remote


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();publish(a.output)
