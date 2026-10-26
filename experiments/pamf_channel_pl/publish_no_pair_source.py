"""Publish the independent clean/H dual-L/U no-peer source pilot."""
import argparse
from . import no_pair_source as source
from .publish_labeled_h_source import publish

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args();publish(a.repo,a.output,a.context,source_module=source)
