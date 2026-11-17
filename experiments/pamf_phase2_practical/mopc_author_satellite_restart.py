"""Build an author-version MoPC-only satellite successor without duplicating A0."""
import argparse
from collections import Counter
from pathlib import Path, PurePosixPath

from .io import read, require, write
from .satellite_native_da import RECEIVERS

POINTS = [[5, 5], [5, 10], [10, 5], [10, 10]]
PROFILES = ('mopc_hr', 'protonet_cda', 'mrior_sda', 'twostage', 'poster_ft', 'tweak_m1')


def make_plan(previous, run_root, release_root):
    root = Path(run_root)
    require(PurePosixPath(run_root).is_absolute() and not root.exists(), 'Run root must be new and absolute')
    release = Path(release_root)
    require(PurePosixPath(release_root).is_absolute(), 'Release root must be absolute')
    require(previous.get('entry_module') == 'satellite_mix3' and
            previous.get('input_condition') == 'satellite_received', 'Wrong predecessor')
    selected = []
    for old in previous['jobs']:
        is_native = old.get('group') == 'native' and old.get('profile') == 'mopc_hr'
        is_hybrid = old.get('group') == 'hybrid' and old.get('profile') in PROFILES[1:] and old.get('methods') == ['mopc_hr']
        if not (is_native or is_hybrid):
            continue
        job = dict(old)
        require(job.get('seed') in (2026092701, 2026092702, 2026092703), 'Wrong model seed')
        cap = job.get('capsule', '').split('/')
        require(len(cap) == 4 and cap[0] == 'capsules' and cap[1] in ('P01', 'P02', 'P03') and
                cap[2] in ('MIX01', 'MIX02', 'MIX03') and cap[3] in RECEIVERS, 'Wrong capsule')
        name = f"{job['group']}-{job['profile']}-s{job['seed']}-{'-'.join(cap)}"
        job.update(id=name, output=str(PurePosixPath(run_root)/'predictions'/name), score=str(PurePosixPath(run_root)/'scores'/name),
                   mopc_stabilization='none')
        if is_native:
            job.update(design='mix3_mopc_author_c_only_v1', satellite_a0_ks=[], satellite_c_points=POINTS)
        else:
            require(job.get('points') == POINTS and job.get('satellite_hybrid') is True and
                    job.get('save_B_for_mopc') is True, 'Wrong hybrid job')
            if job['profile'] == 'protonet_cda':
                require(job.get('mopc_bridge') == 'proto_affine_v1', 'Wrong ProtoNet bridge')
        selected.append(job)
    counts = Counter(j['profile'] for j in selected)
    require(counts == Counter({p:189 for p in PROFILES}) and len({j['id'] for j in selected}) == 1134,
            'Incomplete or duplicate MoPC matrix')
    out = {k: v for k, v in previous.items() if k != 'jobs'}
    out.update(run_root=str(PurePosixPath(run_root)), release_root=str(PurePosixPath(release_root)), jobs=selected,
               allowed_gpus=list(range(8)), max_per_gpu=6, minimum_free_MiB=6000)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', required=True)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--release-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    write(args.output, make_plan(read(args.previous), args.run_root, args.release_root))


if __name__ == '__main__':
    main()
