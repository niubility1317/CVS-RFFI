"""Observe existing full loaders; no filtering or selection."""
class RecordedLoader:
    def __init__(self,loader,role,record,steps=None):
        self.original=loader;self.role=role;self.record=record;self.steps=steps
    def __getattr__(self,name):return getattr(self.original,name)
    def __len__(self):return len(self.original) if self.steps is None else self.steps
    def __iter__(self):
        for batch in self.original:
            self.record(self.role,batch)
            yield batch
