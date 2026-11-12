"""Run one bound clean external task; this is not a GPU queue or scoring tool."""
import argparse
from pathlib import PurePosixPath

from .io import read, require


NATIVE_DA = {'protonet_cda','mrior_sda','dadda_sda','twostage','poster_ft'}
NATIVE_CIL = {'orthogonal','lora_triplet_knn'}
HYBRID_B = NATIVE_DA
HYBRID_C = {'orthogonal','lora'}


def validate(rt, job, group):
    require(rt.get('clean_root') and rt.get('data_root'), 'Clean and parent data roots required')
    require(job.get('seed') in (2026092701,2026092702,2026092703,2026092704), 'Unexpected seed')
    parts=PurePosixPath(job.get('capsule','')).parts
    require(len(parts)==4 and parts[0]=='capsules' and all(p not in ('.','..') for p in parts), 'Invalid task capsule')
    require(job.get('output'), 'Immutable output path required')
    require(job.get('scope') is None, 'Explicit clean workpoints must not be overridden by a legacy scope')
    points=job.get('points',[])
    require(points and len({tuple(p) for p in points})==len(points) and
            all(k in (1,5,10,20) and n in (2,5,10,20) for k,n in points),'Invalid workpoints')
    if group=='native_da':
        require(job.get('profile') in NATIVE_DA, 'Native DA profile unbound or excluded')
    elif group=='native_cil':
        require(job.get('profile') in NATIVE_CIL, 'Native CIL profile unbound or unresolved')
    elif group=='hybrid':
        methods=job.get('methods',[])
        require(job.get('profile') in HYBRID_B and methods and len(set(methods))==len(methods)
                and set(methods)<=HYBRID_C, 'Hybrid bridge unbound or unresolved')
    else:
        raise ValueError('Unknown clean external group')


def run(rt,job,group):
    validate(rt,job,group)
    if group=='hybrid':
        from .group2_worker import predict
    else:
        from .group1_target import predict
    predict(rt,job)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',required=True);p.add_argument('--job',required=True)
    p.add_argument('--group',required=True,choices=['native_da','native_cil','hybrid'])
    args=p.parse_args();run(read(args.runtime),read(args.job),args.group)


if __name__=='__main__':main()
