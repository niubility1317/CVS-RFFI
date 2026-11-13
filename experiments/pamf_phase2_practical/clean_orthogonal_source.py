"""Source-only Orthogonal preparation; never dispatches target predictions."""
import argparse
import os
from pathlib import Path

from .group1 import SEEDS, validate_source_config
from .io import read, require


def check(config, seed=None):
    validate_source_config(config)
    require(seed is None or seed in SEEDS,'Unexpected model seed')
    for name in ('run_root','release_root','source_cache','source_contract','augmentation_cache'):
        require(bool(config.get(name)),f'Missing {name}')
    require(Path(config['release_root']).is_dir() and Path(config['source_cache']).is_dir()
            and Path(config['source_contract']).is_file(),'Missing source inputs')
    root=Path(config['run_root']).resolve()
    cache=Path(config['augmentation_cache']).resolve()
    reuse=config.get('reuse_source_augmentation',False)
    if reuse:
        from experiments.practical_full_2x2.cache import key, probability
        count=len(read(config['source_contract'])['role_ids']['L_s'])
        for epoch in range(1,201):
            p=probability((epoch-1)/200)
            folder=cache/'F0'/key(epoch,1,p)
            marker=read(folder/'complete.json')
            request=marker['request']
            require(marker['status']=='L_AUGMENTATION_READY' and marker['count']==count
                and marker['U_reads']==marker['V_reads']==marker['target_reads']==0
                and marker['dtype']=='float32' and marker['extra_normalization'] is False
                and (request['protocol_id'],request['epoch'],request['stage'],request['probability'])==('F0',epoch,1,p)
                and Path(request['source_cache']).resolve()==Path(config['source_cache']).resolve()
                and Path(request['source_contract']).resolve()==Path(config['source_contract']).resolve()
                and all((folder/name).is_file() for name in ('ids.npy','x.npy','scene.npy')),
                f'Source augmentation cache mismatch at epoch {epoch}')
    else:
        require(cache.is_relative_to(root) and cache!=root,'Source augmentation must use this run root')
    if seed is not None:
        require(not (root/'source'/f'orthogonal-s{seed}').exists(),'Source output already exists')
    return dict(profile='orthogonal',seed=seed,epochs=200,reused_augmentation_epochs=200 if reuse else 0,
                source_only=True,target_reads=0,query_reads=0)


def train(config,seed):
    check(config,seed)
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','')
    require(physical.isdigit() and 0<=int(physical)<=7,'Single physical GPU required')
    from .runner import occupancy
    device=occupancy()[int(physical)]
    require(device['free']>=6000 and not (device['pids']-{os.getpid()}) and not device.get('blocked'),
            'GPU is occupied or has insufficient free memory')
    from .group1_source import train as train_source
    out=Path(config['run_root'])/'source'/f'orthogonal-s{seed}'
    train_source(config,'orthogonal',seed,out,device='cuda:0')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('check','service','train'))
    p.add_argument('--config',required=True);p.add_argument('--seed',type=int)
    args=p.parse_args();config=read(args.config)
    if args.action=='check':print(check(config,args.seed))
    elif args.action=='service':
        check(config)
        require(not config.get('reuse_source_augmentation'),'Reusable augmentation cache is read-only')
        from experiments.practical_full_2x2.cache import service
        service(config)
    else:
        require(args.seed is not None,'Seed required')
        train(config,args.seed)


if __name__=='__main__':main()
