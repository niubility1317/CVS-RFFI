"""Bounded synthetic cache benchmark; does not load datasets or checkpoints."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import cvsrffi.practical_adapter as adapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    x = torch.randn(args.batch_size, 2, 256, generator=torch.Generator().manual_seed(41))
    ids = [f'synthetic-{i}' for i in range(len(x))]
    cases = []
    with tempfile.TemporaryDirectory(prefix='practical-cache-bench-') as cache:
        for route, eq, method in [('full',False,'zf'), ('full',True,'zf'), ('full',True,'mmse'), ('residual',False,'zf')]:
            config = SimpleNamespace(practical_route=route, practical_equalization=eq,
                practical_equalizer_method=method, practical_fs_hz=25e6, practical_fc_hz=2.462e9,
                practical_receiver_seed=2027, practical_eval_cache_dir='')
            for scene in adapter.PRACTICAL[1:]:
                adapter.set_evaluation_context(ids)
                def measure(directory):
                    config.practical_eval_cache_dir=directory
                    started=time.perf_counter()
                    y,_=adapter.apply_practical(x,scene,config,gen=torch.Generator().manual_seed(7))
                    return time.perf_counter()-started,y,adapter._last_meta['cache_event']
                uncached=[]
                for _ in range(args.repeats):
                    sec,reference,_=measure('');uncached.append(sec)
                cold,y,event=measure(cache)
                assert event=='miss' and torch.equal(reference,y)
                warm=[]
                for _ in range(args.repeats):
                    sec,y,event=measure(cache);warm.append(sec)
                    assert event=='hit' and torch.equal(reference,y)
                cases.append(dict(route=route,equalization=eq,method=method,scene=scene,
                    uncached_median_s=statistics.median(uncached),cold_cache_s=cold,
                    warm_cache_median_s=statistics.median(warm),
                    warm_speedup=statistics.median(uncached)/statistics.median(warm),exact_iq=True))
        cache_bytes=sum(p.stat().st_size for p in Path(cache).rglob('*.npz'))
    result=dict(scope='CPU synthetic IQ generation/cache only; not end-to-end training speedup',
                batch_size=args.batch_size,repeats=args.repeats,torch_version=torch.__version__,
                cache_bytes=cache_bytes,cases=cases)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
