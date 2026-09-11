"""Collect three completed A6 nested source runs; no training or target access."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.anchored_aggregation import aggregate_source_runs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'configs/core90_anchored_geometry_v1.json'))
    parser.add_argument('--inputs',nargs=3,required=True,metavar='A6_FUSE_DIRECTORY')
    parser.add_argument('--contract',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    verdict=aggregate_source_runs(args.inputs,args.output,json.loads(Path(args.config).read_text(encoding='utf-8')),args.contract)
    print(json.dumps(dict(status=verdict['status'],passed=verdict['passed'],additional_fits=0,output=str(Path(args.output).resolve()))))


if __name__=='__main__':main()
