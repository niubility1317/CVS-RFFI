"""Publish one source-only nonparametric diagnostic on the original E20."""
import argparse
from . import rx_consensus_source as source
from .publish_labeled_h_source import publish

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args();publish(a.repo,a.output,a.context,source_module=source)
