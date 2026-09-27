"""Hide U_s TX truth at the batch boundary while retaining lineage metadata."""


class UnlabeledSourceView:
    """Only received IQ and legal temporal/domain bookkeeping reach training.

    The physical index remains accessible to the checkpoint contract checker;
    it is never collated into the auxiliary task or interpreted as a TX label.
    """
    _METADATA = frozenset({'rx_i', 'day_i', 'eq_i', 'sig_i', 'base_index', 'split_source'})

    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        iq, _, domain, original = self.dataset[index]
        metadata = {key:value for key,value in original.items() if key in self._METADATA}
        metadata['tx_label_visible'] = False
        return iq, -1, domain, metadata
