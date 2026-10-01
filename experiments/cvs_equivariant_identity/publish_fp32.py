"""New immutable release/run; reuse the existing four-row source dispatcher."""
import argparse
from pathlib import Path
from experiments.cvs_equivariant_identity import publish as original
from experiments.cvs_equivariant_identity.prepare_fp32 import RUN,RELEASE


if __name__=='__main__':
    original.RUN=RUN;original.RELEASE=RELEASE
    original.REMOTE=original.REMOTE.replace('experiments/cvs_equivariant_identity/configs/launch_spec.json','experiments/cvs_equivariant_identity/configs_fp32/launch_spec.json')
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args()
    if args.inspect:original.inspect(args.output)
    else:original.publish(args.output)
