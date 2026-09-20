"""Opt-in disk cache for deterministic evaluation IQ, never training views."""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np

FIXED_NAMESPACES = frozenset(('target_fixed_v3', 'source_validation_fixed_v3'))


@lru_cache(maxsize=1)
def _implementation_id():
    import leo_practical
    root = Path(leo_practical.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cached_evaluation_batch(array, cfg, *, cache_dir, compute, seed, sample_ids,
                            session_ids, realization_namespace, receiver_seed):
    """Return the normal batch result and a cache event.

    Key includes the exact preprocessed input, ordered IDs, RNG draw and channel
    implementation. Caller draws the RNG seed even on hits. No checkpoint or
    labels enter the cache. Atomic writes permit concurrent equivalent readers.
    Cache files contain arrays and JSON only, never pickled Python objects.
    """
    kwargs = dict(seed=seed, sample_ids=sample_ids, session_ids=session_ids,
                  realization_namespace=realization_namespace,
                  receiver_seed=receiver_seed, return_meta=True)
    if not cache_dir or realization_namespace not in FIXED_NAMESPACES:
        return compute(array, cfg, **kwargs), 'bypass'
    header = dict(schema=2, implementation=_implementation_id(), numpy=np.__version__,
                  config=asdict(cfg), seed=int(seed), receiver_seed=int(receiver_seed),
                  ids=list(sample_ids), sessions=list(session_ids),
                  namespace=realization_namespace, dtype=array.dtype.str, shape=array.shape)
    digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    key = digest.hexdigest()
    path = Path(cache_dir) / key[:2] / (key + '.npz')
    event = 'miss'
    try:
        if path.exists():
            with np.load(path, allow_pickle=False) as entry:
                y, states = entry['iq'], entry['states']
                records = json.loads(entry['metadata'].tobytes().decode('utf-8'))
                required = {'sample_id', 'quality_snr_db', 'output_frequency_hz',
                            'channel_equalization_applied', 'geometry'}
                valid_records = isinstance(records, list) and all(
                    isinstance(record, dict) and required.issubset(record)
                    and isinstance(record['geometry'], dict)
                    and {'elevation_deg', 'altitude_m'}.issubset(record['geometry'])
                    for record in records)
                if (str(entry['key'].item()) != key or y.shape != array.shape
                        or states.shape != (len(array),) or not valid_records
                        or len(records) != len(array)
                        or not np.isfinite(y).all()):
                    raise ValueError('invalid practical cache entry')
                return (y, records, states), 'hit'
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        event = 'invalid_recomputed'
    result = compute(array, cfg, **kwargs)
    y, records, states = result
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.writing', delete=False) as f:
            temporary = f.name
            metadata = np.frombuffer(json.dumps(records, separators=(',', ':')).encode('utf-8'), dtype=np.uint8)
            np.savez(f, iq=y, states=states, metadata=metadata, key=key)
        os.replace(temporary, path)
    except OSError:
        # A full/read-only cache must not prevent the original computation.
        event = 'write_unavailable'
    finally:
        if temporary and os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass  # Cache cleanup cannot invalidate already computed IQ.
    return result, event
