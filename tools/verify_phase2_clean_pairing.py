"""Independent read-only audit of Phase2 clean identity and IQ alignment."""
import json
from pathlib import Path

import numpy as np

BASE = Path('/home/szu2070436088/2510044040/CV-SincNet/runs')
RAW = BASE/'20261029-phase2-fullpool-episodes-s2026102901-r01'/'builder_only'/'pool'
PARENT = BASE/'20261029-phase2-threepool-episodes-s2026102901-r01'
OUTPUT = BASE/'20261112-phase2-cleanpaired-data-r01'
RXS = ('1-1', '14-7', '2-1', '20-1', '7-14', '7-7', '8-8')


def main():
    complete = json.loads((OUTPUT/'completion.json').read_text())
    if complete['status'] != 'CLEAN_PAIRED_COMPLETE' or complete['capsules'] != 483 or complete['splits'] != 9660:
        raise ValueError('Incomplete output')
    sample_count = capsules = split_count = 0
    for rx in RXS:
        with np.load(RAW/rx/'records.npz', allow_pickle=False) as f:
            raw_ids = f['physical_ids']
        with np.load(PARENT/'builder_only'/'partition'/rx/'partition.npz', allow_pickle=False) as f:
            partition = f['pool_indices']
            if not np.array_equal(f['physical_ids'], raw_ids):
                raise ValueError('Different source/partition ID order')
        raw = np.load(RAW/rx/'raw_iq.npy', mmap_mode='r', allow_pickle=False)
        clean = np.load(OUTPUT/'banks'/rx/'clean.npy', mmap_mode='r', allow_pickle=False)
        if raw.shape != clean.shape or clean.dtype != np.float32:
            raise ValueError('Incorrect clean bank')
        # Scalar calculation follows the original complex128 routine separately.
        positions = np.random.default_rng(20261112).choice(len(raw), size=128, replace=False)
        for i in positions:
            z = raw[i, 0].astype(np.float64) + 1j*raw[i, 1].astype(np.float64)
            z = z / np.sqrt(np.mean(np.abs(z)**2))
            if not np.array_equal(clean[i], np.stack((z.real, z.imag)).astype(np.float32)):
                raise ValueError('Clean normalization mismatch')
            sample_count += 1
        for pool_index in range(3):
            pool = f'P{pool_index+1:02d}'
            for episode in sorted((PARENT/'capsules'/pool).iterdir()):
                cap = episode/rx
                if not cap.is_dir():
                    continue
                mapping = np.load(OUTPUT/'capsules'/pool/episode.name/rx/'row_to_clean.npy', allow_pickle=False)
                ids = np.load(cap/'ids.npy', allow_pickle=False)
                with np.load(PARENT/'score_only'/pool/episode.name/(rx+'.npz'), allow_pickle=False) as f:
                    if not np.array_equal(ids, f['observation_ids']):
                        raise ValueError('Wrong observation IDs')
                    if not np.array_equal(raw_ids[mapping], f['physical_ids']):
                        raise ValueError('Wrong physical pairing')
                if len(np.unique(mapping)) != len(mapping) or not (partition[mapping] == pool_index).all():
                    raise ValueError('Duplicate or cross-pool record')
                if len(mapping) != int((partition == pool_index).sum()):
                    raise ValueError('Incomplete pool')
                manifest = json.loads((cap/'manifest.json').read_text())
                seen = set()
                for path in (cap/'splits').glob('*.json'):
                    split = json.loads(path.read_text())
                    if path.stem != split['split_id'] or split['capsule_id'] != manifest['capsule_id']:
                        raise ValueError('Split filename or capsule identity mismatch')
                    pair = (split['k'], split['new_count'])
                    if pair in seen:
                        raise ValueError('Duplicate split')
                    seen.add(pair)
                    registered = np.load(cap/'stage_rows'/f"N{split['new_count']:02d}.npy", allow_pickle=False)
                    support = np.asarray(split['support_indices'])
                    if len(support) != split['k']*(6+split['new_count']) or not np.isin(support, registered).all():
                        raise ValueError('Invalid support')
                    if len(registered)-len(support) != split['query_count']:
                        raise ValueError('Invalid query complement')
                    split_count += 1
                if seen != {(k,n) for k in (1,5,10,20) for n in (0,2,5,10,20)}:
                    raise ValueError('Incomplete K/N splits')
                capsules += 1
    if capsules != 483 or split_count != 9660:
        raise ValueError('Missing capsule mappings or splits')
    print(json.dumps({'status':'VERIFIED', 'capsules':capsules, 'physical_records':complete['physical_records'],
                      'splits':split_count, 'independent_scalar_iq_samples':sample_count, 'query_labels_read':False,
                      'predictions_or_training':False}))


if __name__ == '__main__':
    main()
