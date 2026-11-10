"""Expose a role-local record equality key without revealing hidden U labels."""
from contextlib import contextmanager


def indexed_dataset_type(base):
    class IndexedSource(base):
        def __getitem__(self, index):
            x, label, domain, metadata = super().__getitem__(index)
            metadata = dict(metadata)
            if 'branch_record_id' in metadata:
                raise ValueError('Unexpected source record ID collision')
            metadata['branch_record_id'] = int(index)
            return x, label, domain, metadata
    return IndexedSource


@contextmanager
def indexed_source_scope():
    from experiments.cric_dura_comparison import f_data
    previous = f_data.SourceDataset
    f_data.SourceDataset = indexed_dataset_type(previous)
    try:
        yield
    finally:
        f_data.SourceDataset = previous
