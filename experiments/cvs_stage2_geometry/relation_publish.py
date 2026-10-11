"""Publish the fixed source-only relation audit through the CPU release path."""
import argparse
from .conditional_publish import publish,inspect
from .relation_audit import RUN,RELEASE

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args()
    if a.inspect:inspect(a.output)
    else:publish(a.output,run_id=RUN,release_name=RELEASE,
                 module='experiments.cvs_stage2_geometry.relation_audit',
                 checks_module='experiments.cvs_stage2_geometry.relation_checks')
