"""Explicit source-only stages for CORE90 anchored geometry v1."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.anchored_pipeline import STAGES,CANDIDATES,validate_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'configs/core90_anchored_geometry_v1.json'))
    parser.add_argument('--validate-config',action='store_true')
    parser.add_argument('--stage',choices=STAGES)
    parser.add_argument('--candidate',choices=CANDIDATES,default='A4')
    parser.add_argument('--seed',type=int,default=392005)
    parser.add_argument('--role',choices=('L_s','V'),default='L_s')
    for name in ('ground','contract','source','cache','input','output'):parser.add_argument('--'+name)
    parser.add_argument('--device',default='cpu');parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();config=validate_config(json.loads(Path(args.config).read_text(encoding='utf-8')))
    if args.validate_config:
        print(json.dumps(dict(status='VALID',schema=config['schema'],stages=config['stages'],source_only=True)));return
    required=['stage','ground','contract','output']
    if args.stage in {'cache','profile'}:required.append('source')
    if args.stage in {'fit','oof','fuse','calibrate'}:required.append('cache')
    if args.stage in {'fuse','calibrate','export','profile'}:required.append('input')
    if any(getattr(args,k) is None for k in required):parser.error('required arguments: '+', '.join('--'+k for k in required))
    if args.threads<1:parser.error('--threads must be positive')
    import torch
    torch.set_num_threads(args.threads)
    from cvsrffi.anchored_source import run_stage
    run_stage(args,config)
    print(json.dumps(dict(status='SOURCE_STAGE_COMPLETE',stage=args.stage,output=str(Path(args.output).resolve()))))


if __name__=='__main__':main()
