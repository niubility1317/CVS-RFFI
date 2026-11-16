"""Record unlabeled pseudo decisions without opening unlabeled truth.

The file is an observation artifact, not an input to training or selection.
It preserves each repeated exposure rather than claiming unique coverage.
"""
import gzip
import json
from pathlib import Path


class PseudoObservationWriter:
    def __init__(self,path,dataset):
        self.path=Path(path)
        metadata=dataset.metadata
        if dataset.role!='U_s' or 'y' in metadata:
            raise ValueError('Only the original TX-blind U dataset is permitted')
        indices=metadata['base_index'].tolist()
        ids=metadata['ids'].tolist()
        if len(indices)!=len(ids) or len(set(indices))!=len(indices) or len(set(ids))!=len(ids):
            raise ValueError('Unique U source index and physical ID mapping required')
        self.ids=dict(zip((int(i) for i in indices),(str(i) for i in ids)))
        self.file=gzip.open(self.path,'xt',encoding='utf-8',compresslevel=3)
        self.calls=self.seen=self.accepted=0

    def record(self,*,step,batch,pseudo,confidence,accepted):
        import torch
        if (not isinstance(step,int) or step<=0 or not bool((batch[1]==-1).all())
                or pseudo.ndim!=1 or confidence.shape!=pseudo.shape or accepted.shape!=pseudo.shape
                or accepted.dtype!=torch.bool or pseudo.dtype!=torch.long
                or not bool(torch.isfinite(confidence).all())):
            raise ValueError('Blind U batch and aligned finite pseudo decisions required')
        indices=batch[3]['base_index'].tolist()
        if len(indices)!=len(pseudo):raise ValueError('U physical IDs must align with decisions')
        try:ids=[self.ids[int(i)] for i in indices]
        except KeyError as error:raise ValueError('U physical ID not in registered source role') from error
        row=dict(step=step,physical_ids=ids,pseudo=pseudo.detach().cpu().tolist(),
                 confidence=confidence.detach().cpu().tolist(),accepted=accepted.detach().cpu().tolist())
        self.file.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
        self.calls+=1;self.seen+=len(ids);self.accepted+=int(accepted.sum())

    def close(self):
        self.file.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


class PseudoObservationMixin:
    """Opt-in mixin for future REPL objectives; no predictions alter a loss."""
    def validate_unlabeled_batch(self,batch):
        super().validate_unlabeled_batch(batch)
        if getattr(self,'_observation_batch',None) is not None:
            raise ValueError('Previous U observation was not consumed')
        self._observation_batch=batch

    def accept_pseudo(self,student,x,confidence,pseudo):
        accepted=super().accept_pseudo(student,x,confidence,pseudo)
        batch=getattr(self,'_observation_batch',None)
        if batch is None:raise ValueError('Observed decisions require their original blind U batch')
        writer=getattr(self,'_pseudo_observation_writer',None)
        if writer is None:
            writer=PseudoObservationWriter(Path(self.c['output_root'])/'U_pseudo_observations.jsonl.gz',
                                           student._cd_U_loader.dataset)
            self._pseudo_observation_writer=writer
        writer.record(step=int(self.c.get('step_offset',0))+self.calls+1,batch=batch,pseudo=pseudo,
                      confidence=confidence,accepted=accepted)
        self._observation_batch=None
        return accepted

    def close_pseudo_observation(self):
        writer=getattr(self,'_pseudo_observation_writer',None)
        if writer is None:return dict(calls=0,seen=0,accepted=0)
        writer.close()
        return dict(calls=writer.calls,seen=writer.seen,accepted=writer.accepted)
