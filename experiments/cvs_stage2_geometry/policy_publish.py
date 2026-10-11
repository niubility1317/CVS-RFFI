import argparse
from .conditional_publish import publish,inspect
from .policy_audit import RUN,RELEASE

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--inspect',action='store_true');args=p.parse_args()
    if args.inspect:inspect(args.output)
    else:publish(args.output,run_id=RUN,release_name=RELEASE,module='experiments.cvs_stage2_geometry.policy_audit',checks_module='experiments.cvs_stage2_geometry.policy_checks')
