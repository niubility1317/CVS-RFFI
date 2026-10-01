import argparse
from pathlib import Path
from experiments.cvs_equivariant_identity.collect import collect
from experiments.cvs_equivariant_identity.prepare_fp32 import RUN

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();collect(args.root,args.output,run_id=RUN)
