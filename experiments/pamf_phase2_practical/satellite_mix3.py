"""Revised three-seed satellite comparison with a single shared GPU owner."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from .io import read, require, write
from .satellite_native_da import RECEIVERS, SEEDS

POINTS = [[k, n] for k in (5, 10) for n in (5, 10)]
HYBRIDS = ('protonet_cda', 'mrior_sda', 'twostage', 'poster_ft', 'tweak_m1')
NATIVE = ('orthogonal', 'lora_triplet_knn', 'mopc_hr')


def prepare(inventory, a0_inventory, sources, root, release, config_path, wait_run):
    from .satellite_native_c import validate_scope
    from .satellite_native_da import validate_target
    require(inventory.get('status') == 'VERIFIED' and not inventory.get('issues'), 'Invalid DA reuse inventory')
    require(a0_inventory.get('status') == 'VERIFIED' and not a0_inventory.get('issues'), 'Invalid A0 reuse inventory')
    root, release, config_path = Path(root).resolve(), Path(release).resolve(), Path(config_path).resolve()
    require(not root.exists() and release.is_dir() and not config_path.parent.exists(), 'Existing root or release absent')
    config_path.parent.mkdir(parents=True)
    runtime_files = {}
    for key, source in sources.items():
        rt = read(source)
        rt.pop('clean_root', None); rt.pop('restored_root', None)
        rt['input_condition'] = 'satellite_received'
        rt['release_root'] = str(release)
        rt['runtime_file'] = str(config_path.parent / (key + '-runtime.json'))
        require(Path(rt['data_root']).is_dir() and Path(rt['source_contract']).is_file()
                and Path(rt['source_cache']).is_dir(), 'Missing source or data handles')
        write(rt['runtime_file'], rt); runtime_files[key] = rt['runtime_file']
    available = {p: {(v['capsule'], v['seed'], v['k']) for v in a0_inventory['existing'][p]}
                 for p in ('lora_triplet_knn', 'mopc_hr')}
    jobs = []
    def append(group, profile, cap, seed, **kwargs):
        ident = group + '-' + profile + '-s' + str(seed) + '-' + cap.replace('/', '-')
        jobs.append(dict(id=ident, group=group, profile=profile, capsule=cap, seed=seed,
                         output=str(root/'predictions'/ident), score=str(root/'scores'/ident), **kwargs))
    for profile in ('mrior_sda', 'twostage', 'poster_ft'):
        grouped = defaultdict(list)
        for item in inventory['profiles'][profile]['missing_pairs']:
            grouped[(item['capsule'], item['seed'])].append(item['k'])
        missing = {(cap, seed, stage, k, 0) for (cap, seed), ks in grouped.items()
                   for k in ks for stage in ('A0', 'B')}
        for (cap, seed), ks in sorted(grouped.items()):
            job = dict(profile=profile, capsule=cap, seed=seed, satellite_ks=sorted(ks))
            validate_target(job, missing)
            require(cap.split('/')[2] in ('MIX01','MIX02','MIX03') and seed in SEEDS[:3], 'DA outside matrix')
            append('da', profile, cap, seed, satellite_ks=sorted(ks))
    require(not inventory['profiles']['protonet_cda']['missing_pairs'], 'ProtoNet missing tasks need reconciliation')
    for pool in ('P01','P02','P03'):
        for scene in ('MIX01','MIX02','MIX03'):
            for receiver in RECEIVERS:
                cap = f'capsules/{pool}/{scene}/{receiver}'
                for seed in SEEDS[:3]:
                    for profile in NATIVE:
                        a0 = [5,10] if profile == 'orthogonal' else ([] if scene == 'MIX01' else [10])
                        if profile != 'orthogonal':
                            require(all((cap, seed, k) in available[profile] for k in (5,10) if k not in a0),
                                    'Existing native A0 unavailable')
                        extra = dict(design='mix3_v1', satellite_c_points=POINTS, satellite_a0_ks=a0)
                        append('native', profile, cap, seed, **extra); validate_scope(jobs[-1])
                    append('tweak', 'tweak_m1', cap, seed, design='mix3_v1',
                           points=[[k,5] for k in (1,5,10,20)])
                    for profile in HYBRIDS:
                        key = f'{profile}-s{seed}-{cap.replace("/", "-")}'
                        opts = dict(key=key, points=POINTS, methods=['mopc_hr'], satellite_hybrid=True,
                                    save_B_for_mopc=True, mopc_stabilization='none')
                        if profile == 'protonet_cda': opts['mopc_bridge'] = 'proto_affine_v1'
                        append('hybrid', profile, cap, seed, **opts)
    require(len({j['id'] for j in jobs}) == len(jobs), 'Duplicate jobs')
    # Round-robin the groups and profiles so one long group does not starve the others.
    groups = defaultdict(list)
    for job in jobs: groups[(job['group'], job['profile'])].append(job)
    ordered = []
    while any(groups.values()):
        for key in sorted(groups):
            if groups[key]: ordered.append(groups[key].pop())
    for index, job in enumerate(ordered):
        job['id'] = f'{index:05d}-' + job['id']
    return dict(entry_module='satellite_mix3', input_condition='satellite_received', run_root=str(root),
                release_root=str(release), runtimes=runtime_files, wait_run=str(wait_run),
                allowed_gpus=list(range(8)), max_per_gpu=6, minimum_free_MiB=6000,
                worker_reserve_MiB=3000, jobs=ordered)


def verify_prediction(job):
    if job['group'] == 'da':
        from .satellite_native_da import verify_prediction as check
    elif job['group'] == 'native':
        from .satellite_native_c import verify_prediction as check
    elif job['group'] == 'tweak':
        from .satellite_tweak import verify_prediction as check
    else:
        from .group2_worker import method_arm
        from .mopc_identity import require_author_mopc_row
        folder = Path(job['output']); done = read(folder/'complete.json'); idx = read(folder/'predictions.json')
        require(done.get('status') == 'PREDICTIONS_COMPLETE' and done.get('truth_read') is False
                and done.get('query_fit') is False and done.get('profile') == job['profile']
                and done.get('model_seed') == job['seed'] and idx.get('capsule') == job['capsule'],
                'Invalid hybrid completion')
        arm = method_arm(job['profile'], 'mopc_hr', job.get('mopc_bridge','original'), 'none')
        points=job.get('points',POINTS) if job.get('author_pilot') is True or job.get('author_continuation') is True else POINTS
        if job.get('author_pilot') is True:
            require(points==[[5,5]] and job['capsule']=='capsules/P01/MIX01/1-1' and job['seed']==2026092701,
                    'Wrong author pilot')
        if job.get('author_continuation') is True:
            require(points==[[5,10],[10,5],[10,10]] and job['capsule']=='capsules/P01/MIX01/1-1' and
                    job['seed']==2026092701, 'Wrong author continuation')
        expected = {(k,n,arm) for k,n in points}
        rows = idx['rows']; actual = [(r['k'],r['new_count'],r['arm']) for r in rows]
        for row in rows: require_author_mopc_row(row)
        if job.get('mopc_sampling_version') == 'author_drop_last_v1':
            from .mopc_identity import require_author_sampling
            for row in rows: require_author_sampling(row)
        require(len(rows) == len(expected) == done['rows'] and set(actual) == expected
                and all(r['stage']=='C' and r.get('input_condition')=='satellite_received'
                        and (folder/r['prediction']).is_file() for r in rows)
                and all((folder/f'B-K{k}.pt').is_file() for k in {p[0] for p in points}), 'Invalid hybrid rows or B states')
        if job['profile']=='tweak_m1':
            require(all(r.get('query_records_per_decision')==1 for r in rows), 'Tweak query not single-record')
        return
    check(job)


def run_worker(config, job):
    from .vendor_native.comparison_suite.registration import MoPCNumericalError, MoPCSamplingError
    group = job['group']; rt = read(config['runtimes']['hybrid' if group in ('hybrid','tweak') else
                                       job['profile'] if group=='native' else 'da'])
    require(not rt.get('clean_root') and rt['input_condition']=='satellite_received', 'Wrong input')
    try:
        if group == 'da':
            from .satellite_native_da import run_worker as execute
            execute(dict(runtime_file=config['runtimes']['da']), job)
        elif group == 'native':
            from .satellite_native_c import run_worker as execute
            execute(config, job)
        elif group == 'tweak':
            from .satellite_tweak import run_worker as execute
            execute(dict(runtime_file=config['runtimes']['hybrid']), job)
        elif job['profile'] == 'tweak_m1':
            from .clean_tweak_lora_worker import run
            run(rt, job)
        else:
            from .group2_worker import predict
            predict(rt, job)
    except (MoPCNumericalError, MoPCSamplingError) as exc:
        if (job['profile']=='mopc_hr' or job.get('methods')==['mopc_hr']) and Path(job['output']).is_dir():
            marker=Path(job['output'])/'numerical_failure.json'
            if not marker.exists():
                write(marker, dict(status='MOPC_SAMPLING_FAILURE' if isinstance(exc, MoPCSamplingError) else 'MOPC_NUMERICAL_FAILURE',
                      profile=job['profile'], capsule=job['capsule'], seed=job['seed'], error=str(exc),
                      truth_read=False, query_fit=False))
        raise


def run_score(config, job):
    from .episodic_score import run
    verify_prediction(job)
    key = 'hybrid' if job['group'] in ('hybrid','tweak') else job['profile'] if job['group']=='native' else 'da'
    run(read(config['runtimes'][key]), dict(prediction=job['output'], output=job['score']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare','dispatch','worker','score'))
    p.add_argument('--config', required=True); p.add_argument('--job-id')
    p.add_argument('--inventory'); p.add_argument('--a0-inventory'); p.add_argument('--sources')
    p.add_argument('--run-root'); p.add_argument('--release-root'); p.add_argument('--wait-run')
    args = p.parse_args()
    if args.action == 'prepare':
        config = prepare(read(args.inventory), read(args.a0_inventory), read(args.sources),
                         args.run_root, args.release_root, args.config, args.wait_run)
        write(args.config, config)
        print(json.dumps(dict(status='PREPARED', jobs=len(config['jobs'])))); return
    config = read(args.config)
    if args.action == 'dispatch':
        from .satellite_native_c import dispatch
        dispatch(config, args.config); return
    jobs = [j for j in config['jobs'] if j['id'] == args.job_id]
    require(len(jobs)==1, 'Unknown job')
    (run_worker if args.action=='worker' else run_score)(config,jobs[0])


if __name__ == '__main__':
    main()
