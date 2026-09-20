"""One CPU task ahead; consumer exceptions also drain/close the producer."""
from concurrent.futures import ThreadPoolExecutor


def prefetch_map(transform, iterable):
    iterator=iter(iterable)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='practical-view') as pool:
        try:
            pending=pool.submit(transform, next(iterator))
        except StopIteration:
            return
        while True:
            value=pending.result()
            try:
                pending=pool.submit(transform, next(iterator))
            except StopIteration:
                yield value
                return
            yield value
