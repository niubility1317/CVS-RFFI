import argparse
from experiments.cvs_stage2_geometry.conditional_publish import publish,inspect
from .realized_audit import RUN,RELEASE

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args()
    if a.inspect:inspect(a.output)
    else:publish(a.output,run_id=RUN,release_name=RELEASE,
        module='experiments.cvs_stage2_stable_rx.realized_audit',
        checks_module='experiments.cvs_stage2_stable_rx.realized_checks',extra_paths=('experiments/cvs_stage2_stable_rx',))
