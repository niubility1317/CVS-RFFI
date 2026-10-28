"""Offline Full WiSig eligibility, allocation and fixed-received data builder.

Legacy compact IQ is compared by exact rows, never by its incompatible indices.
No model, source checkpoint, target fit, prediction, or score is used here.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np

from .data import identity, observe
from .io import read, write, require, validate_split


def rows(x):
    x = np.asarray(x)
    require(x.ndim == 3 and x.shape[1:] == (256, 2) and x.dtype == np.float64,
            'Legacy/Full raw representation mismatch; implicit normalization is forbidden')
    return np.ascontiguousarray(x).reshape(len(x), 512).view(np.dtype((np.void, 4096))).reshape(-1)


def legacy_reference(directory, classes, receivers):
    values = {}
    manytx = {}
    for filename in ('ManySig.pkl', 'ManyTx.pkl', 'ManyRx.pkl', 'SingleDay.pkl'):
        with (Path(directory) / filename).open('rb') as f:
            data = pickle.load(f)
        tx = list(map(str, data['tx_list'])); rx = list(map(str, data['rx_list']))
        days = list(map(str, data['capture_date_list'])); eq = list(data['equalized_list']).index(1)
        for r in set(receivers) & set(rx):
            for c in set(classes) & set(tx):
                for di, day in enumerate(days):
                    x = data['data'][tx.index(c)][rx.index(r)][di][eq]
                    if x is None:
                        continue
                    x = np.asarray(x)
                    if not len(x):
                        continue
                    key = (r, c, day)
                    v = rows(x).copy()
                    values.setdefault(key, []).append(v)
                    if filename == 'ManyTx.pkl':
                        manytx[key] = np.unique(v)
        del data, x
        gc.collect()
        print('LEGACY_REFERENCE', filename, len(values), flush=True)
    return {k: np.unique(np.concatenate(v)) for k, v in values.items()}, manytx


def full_members(directory, days, receivers):
    found = {}
    for path in sorted(Path(directory).glob('*.zip')):
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            for day in days:
                for r in receivers:
                    member = f'pkl_wifi_eq_{day}/dataset_{day}_node{r}.pkl'
                    if member in names:
                        require((r, day) not in found, 'Duplicate Full receiver/date file')
                        found[(r, day)] = (str(path), member)
    require(len(found) == len(days) * len(receivers), 'Missing target receiver/date Full file')
    return found


def balanced_query_counts(capacity, total=500):
    """Integer max-min allocation; class/receiver counts only, no signal quality."""
    require(len(capacity) == 4 and min(capacity) >= 0 and sum(capacity) >= total,
            'Insufficient independent query capacity after legacy exclusion')
    result = [0] * 4
    for _ in range(total):
        choices = [d for d in range(4) if result[d] < capacity[d]]
        d = min(choices, key=lambda j: (result[j], j))
        result[d] += 1
    return result


def quota_matrix(row_totals, column_totals):
    require(sum(row_totals) == sum(column_totals), 'Date/domain quota totals differ')
    ideal = np.outer(row_totals, column_totals) / sum(row_totals)
    matrix = np.floor(ideal).astype(np.int64)
    left_r = np.asarray(row_totals) - matrix.sum(axis=1)
    left_c = np.asarray(column_totals) - matrix.sum(axis=0)
    while left_r.sum():
        choices = [(i,j) for i in range(len(left_r)) for j in range(4) if left_r[i] and left_c[j]]
        i,j = max(choices, key=lambda ij: (ideal[ij]-matrix[ij], -ij[0], -ij[1]))
        matrix[i,j] += 1; left_r[i] -= 1; left_c[j] -= 1
    require(not left_c.any(), 'Date quota rounding failed')
    return matrix.tolist()


def cell_plan(support_candidates, available, seed, query_sizes, support_size=24):
    require(len(available) == len(support_candidates) == 4, 'Expected four dates')
    rng = np.random.default_rng(seed)
    support_dates = balanced_query_counts([len(s) for s in support_candidates], len(query_sizes)*support_size)
    s_matrix = quota_matrix([support_size]*len(query_sizes), support_dates)
    selected = []
    remaining = []
    for d, (s, a) in enumerate(zip(support_candidates, available)):
        require(set(s) <= set(a), 'Support candidates must be eligible observations')
        chosen = np.asarray(s)[rng.permutation(len(s))[:support_dates[d]]].astype(int).tolist()
        selected.append(chosen)
        remaining.append(sorted(set(a) - set(chosen)))
    q = balanced_query_counts([len(a) for a in remaining], sum(query_sizes))
    q_matrix = quota_matrix(query_sizes, q)
    chosen_q = [np.asarray(a)[rng.permutation(len(a))[:n]].astype(int).tolist() for a, n in zip(remaining, q)]
    result = []
    s_pos = [0]*4; q_pos = [0]*4
    for domain, query in enumerate(query_sizes):
        s = []; t = []
        for d in range(4):
            ns, nq = s_matrix[domain][d], q_matrix[domain][d]
            s.extend((d, i) for i in selected[d][s_pos[d]:s_pos[d]+ns]); s_pos[d] += ns
            t.extend((d, i) for i in chosen_q[d][q_pos[d]:q_pos[d]+nq]); q_pos[d] += nq
        require(len(s) == support_size and len(t) == query, 'Allocation total differs')
        result.append(dict(support=s, query=t))
    all_s = [tuple(x) for p in result for x in p['support']]
    all_q = [tuple(x) for p in result for x in p['query']]
    require(len(set(all_s + all_q)) == len(all_s + all_q), 'Physical role/domain overlap')
    return result, dict(support_per_domain_date=s_matrix, query_per_domain_date=q_matrix)


def inventory(rt):
    root = Path(rt['run_root']); root.mkdir(parents=True, exist_ok=False)
    private = root / 'builder_only'; private.mkdir()
    classes = rt['inventory_classes']
    require(not set(rt['source_receivers']) & set(rt['target_receivers']), 'Source/target RX overlap')
    legacy, reserved = legacy_reference(rt['legacy_directory'], classes, rt['target_receivers'])
    members = full_members(rt['raw_directory'], rt['dates'], rt['target_receivers'])
    cells = {}; details = []; matched = Counter()
    for (r, day), (archive, member) in members.items():
        with zipfile.ZipFile(archive) as z:
            data = pickle.loads(z.read(member))
        for c in classes:
            x = data['data'][data['node_list'].index(c)] if c in data['node_list'] else np.empty((0,256,2),np.float64)
            v = rows(x); key = (r, c, day)
            old = legacy.get(key, np.empty(0, dtype=v.dtype))
            reserved_old = reserved.get(key, np.empty(0, dtype=v.dtype))
            require(np.isin(old, v).all(), f'Legacy raw/Full mapping incomplete: {key}')
            require(np.isin(reserved_old, v).all(), f'ManyTx raw/Full mapping incomplete: {key}')
            prior = np.isin(v, old); blocked = np.isin(v, reserved_old)
            _, unique_ix = np.unique(v, return_index=True)
            distinct = np.zeros(len(v),dtype=bool); distinct[unique_ix] = True
            eligible = np.flatnonzero(~blocked & distinct).tolist()
            support = np.flatnonzero(~prior & distinct).tolist()
            cells[key] = dict(available=eligible, support=support)
            details.append(dict(receiver=r, transmitter=c, date=day, full=len(x), exact_distinct=len(unique_ix),
                                legacy_compact_unique=int(prior.sum()), manytx_reserved=int(blocked.sum()),
                                support_candidates=len(support), query_eligible_before_support=len(eligible)))
            matched.update(full=len(x), legacy_compact=int(prior.sum()), reserved_manytx=int(blocked.sum()))
        del data, x, v
        gc.collect()
        print('AUDIT_FULL', r, day, flush=True)
    write(private / 'eligibility.json', dict(cells=[dict(receiver=r,transmitter=c,date=d,**v) for (r,c,d),v in cells.items()],
                                            members=[dict(receiver=r,date=d,archive=a,member=m) for (r,d),(a,m) in members.items()]))
    write(root / 'inventory_report.json',dict(status='RAW_ELIGIBILITY_VERIFIED',cells=details,matched_counts=dict(matched),
                                             historical_policy=rt['historical_policy'],model_reads=0,predictions=0))
    print('INVENTORY_VERIFIED', len(details), flush=True)


def audit(rt):
    from leo_practical.channel import stable_seed
    root=Path(rt['run_root']);private=root/'builder_only'
    inventory=read(private/'eligibility.json')
    cells={(p['receiver'],p['transmitter'],p['date']):p for p in inventory['cells']}
    classes=rt['old_classes']+rt['new_classes']
    require(len(classes)==26 and len(set(classes))==26, 'Require six old and twenty new TX')
    plans = []; quotas = []
    for r in rt['target_receivers']:
        for c in classes:
            s = [cells[(r, c, d)]['support'] for d in rt['dates']]
            a = [cells[(r, c, d)]['available'] for d in rt['dates']]
            p, q = cell_plan(s, a, stable_seed(rt['allocation_seed'], r, c), [p['query_size'] for p in rt['pools']])
            plans.append(dict(receiver=r, transmitter=c, domains=p))
            quotas.append(dict(receiver=r, transmitter=c, **q))
    write(private / 'allocation.json', dict(runtime=rt, members=inventory['members'], cells=plans))
    n_support=len(rt['pools'])*24*7*26;n_query=sum(p['query_size'] for p in rt['pools'])*7*26
    report = dict(status='ALLOCATION_VERIFIED_NO_RECEIVED_YET',
                  historical_policy=rt['historical_policy'], date_quotas=quotas,
                  records=n_support+n_query,support=n_support,query=n_query,
                  supports_exclude_all_four_legacy_compact_sets=True,
                  all_observations_exclude_entire_manytx=True,
                  legacy_raw_exact_mapping_complete=True, model_reads=0, predictions=0)
    write(root / 'allocation_report.json', report)
    print('ALLOCATION_VERIFIED', report['records'], flush=True)


def select_classes(rt):
    report=read(Path(rt['run_root'])/'inventory_report.json')
    details=report['cells'];required=sum(p['query_size']+24 for p in rt['pools'])
    candidates=[]
    for c in rt['inventory_classes']:
        if c in rt['old_classes']:
            continue
        records=[v for v in details if v['transmitter']==c]
        if len(records)!=28 or min(v['full'] for v in records)<=0:
            continue
        min_total=min(sum(v['query_eligible_before_support'] for v in records if v['receiver']==r) for r in rt['target_receivers'])
        min_support=min(sum(v['support_candidates'] for v in records if v['receiver']==r) for r in rt['target_receivers'])
        if min_total<required or min_support<24*len(rt['pools']):
            continue
        candidates.append(dict(transmitter=c,minimum_available_total=min_total,
                               minimum_fresh_support=min_support,minimum_date=min(v['full'] for v in records)))
    candidates.sort(key=lambda v:(-v['minimum_available_total'],-v['minimum_fresh_support'],-v['minimum_date'],v['transmitter']))
    require(len(candidates)>=20, 'Fewer than twenty source-unseen TX meet registered quantity constraints')
    selected=[v['transmitter'] for v in candidates[:20]]
    arrival=np.asarray(selected)[np.random.default_rng(rt['arrival_order_seed']).permutation(20)].tolist()
    write(Path(rt['run_root'])/'selection.json',dict(status='COUNT_ONLY_SELECTION_FIXED',rule=rt['class_selection_rule'],
                                                  selected_coverage_order=selected,arrival_order=arrival,
                                                  eligible_candidates=candidates,required_per_RX_TX=required,
                                                  no_model_or_accuracy_access=True))
    print('CLASS_SELECTION_FIXED',json.dumps(arrival),flush=True)


def build(rt):
    from leo_practical.channel import stable_seed
    root = Path(rt['run_root']); private = root / 'builder_only'
    plan = read(private / 'allocation.json')
    require(plan['runtime'] == rt, 'Do not change allocation/runtime after audit')
    require(read(root / 'allocation_report.json')['status'] == 'ALLOCATION_VERIFIED_NO_RECEIVED_YET', 'Incomplete allocation audit')
    mixture = read(rt['mixture_config']); classes = rt['old_classes'] + rt['new_classes']
    summary=read(root/'allocation_report.json');count=summary['records']
    path = private / 'fixed_received.npy'
    require(not path.exists(), 'Existing received output: reconcile, never overwrite')
    received = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(count, 2, 256))
    ids = []; truth = {}; groups = {}; scene_counts = Counter()
    lookup = {(p['receiver'], p['transmitter']): p for p in plan['cells']}
    n = 0
    for entry in plan['members']:
        r, day = entry['receiver'], entry['date']; di = rt['dates'].index(day)
        with zipfile.ZipFile(entry['archive']) as z:
            data = pickle.loads(z.read(entry['member']))
        for c in classes:
            x = data['data'][data['node_list'].index(c)]
            for pi, pool in enumerate(rt['pools']):
                for role in ('support', 'query'):
                    for day_index, si in lookup[(r, c)]['domains'][pi][role]:
                        if day_index != di:
                            continue
                        sid = identity(['WiSig/Full/eq1', c, r, day, si])
                        require(sid not in truth, 'Repeated raw observation')
                        value, meta = observe(x[si].astype(np.float32).T, mixture['scenarios'], sid=sid,
                                              session=f'rx:{r}/day:{day}', pool=pool, mixture=mixture)
                        require(np.isfinite(value).all(), 'Nonfinite received IQ')
                        received[n] = value; ids.append(sid)
                        truth[sid] = dict(index=n, transmitter=c, receiver=r, day=day, signal_index=si,
                                          pool=pool['id'], role=role, **meta)
                        groups.setdefault((r, pool['id'], c, role), []).append(n)
                        scene_counts[(pool['id'], meta['scenario'])] += 1
                        n += 1
        received.flush()
        del data, x
        gc.collect()
        print('BUILD_RECEIVED', r, day, n, count, flush=True)
    require(n == count, 'Received total mismatch')
    score = root / 'score_only'; score.mkdir()
    write(score / 'truth.json', truth)
    ids = np.asarray(ids, dtype='U24')
    np.save(private / 'ids.npy', ids, allow_pickle=False)
    write(private / 'groups.json', [dict(receiver=r,pool=p,transmitter=c,role=role,indices=ix)
                                  for (r,p,c,role),ix in groups.items()])
    total_splits = 0; catalog = []
    for r in rt['target_receivers']:
        for pool in rt['pools']:
            pid = pool['id']
            seeds = pool.get('support_seeds', rt['mixed_support_seeds'] if 'scenario' not in pool else rt['single_support_seeds'])
            for nc in rt['new_counts']:
                registered = rt['old_classes'] + rt['new_classes'][:nc]
                capsule = root / 'capsules' / r / pid / f'N{nc:02d}'
                capsule.mkdir(parents=True)
                (capsule / 'splits').mkdir(); (capsule / 'supports').mkdir()
                q = [i for c in registered for i in groups[(r, pid, c, 'query')]]
                q = np.asarray(q)[np.random.default_rng(stable_seed(rt['query_order_seed'],r,pid,nc)).permutation(len(q))]
                np.save(capsule / 'query_iq.npy', received[q], allow_pickle=False)
                np.save(capsule / 'query_ids.npy', ids[q], allow_pickle=False)
                capsule_id = rt['run_id'] + ':' + r + ':' + pid + ':' + f'N{nc:02d}'
                row_count = 0
                for ss in seeds:
                    orders = {c: np.asarray(groups[(r,pid,c,'support')])[np.random.default_rng(stable_seed(ss,r,pid,c)).permutation(24)].tolist()
                              for c in registered}
                    for k in rt['shots']:
                        s = [i for c in registered for i in orders[c][:k]]
                        labels = np.repeat(np.arange(len(registered),dtype=np.int64), k)
                        split = dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id=capsule_id,
                                     receiver=r,scenario=pid,k=k,registered_classes=registered,
                                     support_indices=list(range(len(s))),support_labels=labels.tolist(),
                                     query_indices=list(range(len(s),len(s)+len(q))),support_seed=ss)
                        split['split_id'] = identity(split)
                        validate_split(split, {'capsule_id':capsule_id}, len(s)+len(q))
                        require(not set(ids[s]) & set(ids[q]), 'Support/query physical overlap')
                        np.savez(capsule / 'supports' / (split['split_id']+'.npz'), iq=received[s],ids=ids[s],labels=labels)
                        write(capsule / 'splits' / (split['split_id']+'.json'), split)
                        total_splits += 1; row_count += 1
                manifest = dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id=capsule_id,
                                storage='support-plus-shared-query-v1',signal_shape=[2,256],dtype='float32',
                                query_count=len(q),registered_classes=registered,split_count=row_count,
                                query_iq='query_iq.npy',query_ids='query_ids.npy',support_directory='supports',
                                channel=dict(route='residual',mode='post_sync',equalization_enabled=False,fs_hz=25e6))
                write(capsule / 'manifest.json', manifest)
                catalog.append(dict(path=str(capsule.relative_to(root)),capsule_id=capsule_id,receiver=r,pool=pid,
                                    new_count=nc,query_count=len(q),split_count=row_count))
    write(root / 'catalog.json', dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsules=catalog))
    write(root / 'builder_report.json', dict(status='VALIDATED_ONCE',physical_records=n,unique_support_candidates=summary['support'],
                                            unique_query=summary['query'],capsules=len(catalog),splits=total_splits,
                                            realized_scene_counts=[dict(pool=p,scenario=s,count=v) for (p,s),v in scene_counts.items()],
                                            source_roles_unchanged=True,model_reads=0,predictions=0,training_launches=0,
                                            query_labels_absent_from_capsules=True,future_class_inputs_not_in_current_capsule=True,
                                            public_support_files_contain_only_actual_K=True,
                                            historical_policy=rt['historical_policy']))
    print('DATA_COMPLETE', n, len(catalog), total_splits, flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['inventory','select','audit','build'])
    p.add_argument('--runtime',required=True);a=p.parse_args();rt=read(a.runtime)
    import sys
    sys.path.insert(0,str(Path(rt['phase1_release'])/'code'))
    if a.mode in ('audit','build') and rt['new_classes'] is None:
        rt=dict(rt,new_classes=read(Path(rt['run_root'])/'selection.json')['arrival_order'])
    {'inventory':inventory,'select':select_classes,'audit':audit,'build':build}[a.mode](rt)


if __name__=='__main__':
    main()
