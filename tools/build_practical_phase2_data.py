"""Build once: disjoint physical support/query and one residual observation/ID.

This builder runs outside the predictor. Raw paths and query truth remain in
the builder report / score_only directory, never in the runtime capsule.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'code')]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def assignments(records, *, seed, receiver, class_id):
    from leo_practical.channel import stable_seed
    if len(records) < 198:
        raise ValueError('At least198 physical observations required per RX/class')
    if len(set(records)) != len(records):
        raise ValueError('Repeated physical record')
    rng = np.random.default_rng(stable_seed(seed, receiver, class_id, 'scene-role-allocation'))
    order = rng.permutation(len(records))[:198]
    result = {}
    for j, scene in enumerate(('practical_high','practical_mid','practical_low_urban')):
        indices = [records[int(i)] for i in order[j*66:(j+1)*66]]
        result[scene] = {'support_pool': indices[:36], 'query': indices[36:]}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    out = Path(cfg['output_root'])
    out.mkdir(parents=True, exist_ok=False)
    capsule = out / 'capsule'
    capsule.mkdir()
    (capsule / 'splits').mkdir()
    score_only = out / 'score_only'
    score_only.mkdir()
    from dataset_wisig import load_wisig_compact_pkl
    from leo_practical.channel import Config, stable_seed
    from leo_practical.batch import apply_leo_practical_channel_batch
    raw = load_wisig_compact_pkl(cfg['manytx_path'])
    txs = [str(t) for t in raw['tx_list']]
    rxs = [str(r) for r in raw['rx_list']]
    days = [str(d) for d in raw['capture_date_list']]
    old, new = cfg['old_classes'], cfg['new_classes']
    if set(old) & set(new) or len(set(old+new)) != len(old+new):
        raise ValueError('Old/new transmitter identities overlap')
    if set(cfg['target_receivers']) & set(cfg['source_receivers']):
        raise ValueError('Target receivers overlap Phase1 source')
    eq = list(raw['equalized_list']).index(1)
    arrays, ids, ledger, groups, seen = [], [], [], {}, set()
    for receiver in cfg['target_receivers']:
        ri = rxs.index(receiver)
        for class_id in old+new:
            ti = txs.index(class_id)
            records = [(di, si) for di in range(len(days)) if raw['data'][ti][ri][di][eq] is not None
                       for si in range(len(raw['data'][ti][ri][di][eq]))]
            role_map = assignments(records, seed=cfg['data_seed'], receiver=receiver, class_id=class_id)
            for scene, roles in role_map.items():
                channel = Config(fs_hz=cfg['fs_hz'], scenario=scene, processing_route='residual',
                    mode='post_sync', equalization_enabled=False,
                    input_processing_state='WiSig_equalized1_center256_unit_rms')
                for role, selected in roles.items():
                    x, names, sessions = [], [], []
                    for di, si in selected:
                        sid = digest(['WiSig/ManyTx', class_id, receiver, days[di], 1, si])[:24]
                        if sid in seen:
                            raise ValueError('Physical ID reused across scene/role')
                        seen.add(sid)
                        iq = np.asarray(raw['data'][ti][ri][di][eq][si], dtype=np.float32).T
                        if iq.shape != (2,256):
                            raise ValueError('Unexpected IQ shape; no implicit resizing in target builder')
                        x.append(iq)
                        names.append(sid)
                        sessions.append(f'rx:{receiver}/day:{days[di]}')
                    received, metadata, _ = apply_leo_practical_channel_batch(np.stack(x), channel,
                        seed=cfg['augmentation_seed'], sample_ids=names, session_ids=sessions,
                        realization_namespace='phase2/fixed-received/v1', receiver_seed=cfg['receiver_seed'])
                    if not np.isfinite(received).all():
                        raise ValueError('Nonfinite received IQ')
                    idx = list(range(len(ids), len(ids)+len(names)))
                    groups[(receiver, scene, class_id, role)] = idx
                    arrays.extend(received)
                    ids.extend(names)
                    for i, sid, pair, meta in zip(idx, names, selected, metadata):
                        ledger.append(dict(index=i, physical_id=sid, transmitter=class_id,
                            receiver=receiver, day=days[pair[0]], signal_index=pair[1],
                            scene=scene, pool_role=role, old=class_id in old,
                            overlay_seed=int(stable_seed(cfg['augmentation_seed'], 'phase2/fixed-received/v1', scene, sid))))
        print(json.dumps(dict(receiver=receiver, received_records=len(ids))), flush=True)
    all_iq = np.stack(arrays).astype('float32')
    # Numeric storage positions must not encode TX/scene/role blocks.
    order = np.random.default_rng(cfg['data_seed'] + 1).permutation(len(ids))
    inverse = np.empty(len(ids), dtype=np.int64)
    inverse[order] = np.arange(len(ids))
    all_iq = all_iq[order]
    ids = [ids[int(i)] for i in order]
    groups = {key: [int(inverse[i]) for i in values] for key, values in groups.items()}
    for entry in ledger:
        entry['index'] = int(inverse[entry['index']])
    np.savez(capsule/'received.npz', iq=all_iq, ids=np.asarray(ids))
    capsule_id = 'residual-noeq-' + hashlib.sha256((capsule/'received.npz').read_bytes()).hexdigest()[:24]
    split_count = 0
    for receiver in cfg['target_receivers']:
        for scene in cfg['scenarios']:
            for new_count in cfg['new_counts']:
                classes = old + new[:new_count]
                query = [i for c in classes for i in groups[(receiver,scene,c,'query')]]
                rng = np.random.default_rng(stable_seed(cfg['evaluation_seed'], receiver, scene, new_count))
                query = [query[int(i)] for i in rng.permutation(len(query))]
                for support_seed in cfg['support_seeds']:
                    support_order = {}
                    for c in classes:
                        pool = groups[(receiver,scene,c,'support_pool')]
                        rng = np.random.default_rng(stable_seed(support_seed,receiver,scene,c))
                        support_order[c] = [pool[int(i)] for i in rng.permutation(len(pool))]
                    for k in cfg['shots']:
                        support = [i for c in classes for i in support_order[c][:k]]
                        labels = [j for j,c in enumerate(classes) for _ in support_order[c][:k]]
                        if len(set(support)) != k*len(classes) or set(support)&set(query):
                            raise ValueError('Support/query physical-ID validation failed')
                        split = dict(protocol_schema='p2_min_v1', phase2_data_status='VALIDATED_ONCE',
                            capsule_id=capsule_id, receiver=receiver, scenario=scene, k=k,
                            registered_classes=classes, support_indices=support, support_labels=labels,
                            query_indices=query, support_seed=support_seed)
                        split['split_id'] = digest(split)[:24]
                        (capsule/'splits'/f"{split['split_id']}.json").write_text(json.dumps(split),encoding='utf-8')
                        split_count += 1
    manifest = dict(protocol_schema='p2_min_v1', phase2_data_status='VALIDATED_ONCE',
        capsule_id=capsule_id, members=['received.npz','splits/*.json'], received_count=len(ids),
        split_count=split_count, signal_shape=[2,256], scenarios=cfg['scenarios'],
        channel=dict(route='residual',mode='post_sync',equalization_enabled=False,fs_hz=cfg['fs_hz']),
        checks=dict(unique_physical_observation=True, scenes_physically_disjoint=True,
                    support_query_physically_disjoint=True, query_labels_absent=True,
                    source_paths_absent=True, finite_received=True))
    (capsule/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (score_only/'truth.json').write_text(json.dumps({row['physical_id']:row for row in ledger}),encoding='utf-8')
    (out/'builder_report.json').write_text(json.dumps(dict(config=cfg,manifest=manifest,
        physical_record_mapping='WiSig ManyTx TX/RX/day/equalized/signal_index; target RX disjoint from all Phase1 RX',
        old_new_data_family='both target old and new from ManyTx; Phase1 source from ManySig',
        no_checkpoint_access=True),indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)


if __name__ == '__main__':
    main()
