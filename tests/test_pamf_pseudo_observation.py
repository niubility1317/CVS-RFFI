import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from experiments.pamf_repl_weight.pseudo_observation import PseudoObservationWriter, PseudoObservationMixin


def dataset():
    return SimpleNamespace(role='U_s',metadata={'ids':np.array(['a','b']),
                                               'base_index':np.array([4,7])})


class PseudoObservationTests(unittest.TestCase):
    def test_mixin_preserves_acceptance_and_global_step(self):
        class Base:
            def __init__(self,c):self.c=c;self.calls=3
            def validate_unlabeled_batch(self,batch):
                if not bool((batch[1]==-1).all()):raise ValueError('U truth')
            def accept_pseudo(self,student,x,confidence,pseudo):return confidence>=.95
        class Objective(PseudoObservationMixin,Base):pass
        with tempfile.TemporaryDirectory() as folder:
            obj=Objective(dict(output_root=folder,step_offset=4440))
            student=SimpleNamespace(_cd_U_loader=SimpleNamespace(dataset=dataset()))
            batch=(None,torch.tensor([-1,-1]),None,{'base_index':torch.tensor([7,4])})
            confidence=torch.tensor([.97,.90]);pseudo=torch.tensor([1,0])
            obj.validate_unlabeled_batch(batch)
            accepted=obj.accept_pseudo(student,None,confidence,pseudo)
            self.assertTrue(torch.equal(accepted,torch.tensor([True,False])))
            self.assertEqual(obj.close_pseudo_observation(),dict(calls=1,seen=2,accepted=1))
            with gzip.open(Path(folder)/'U_pseudo_observations.jsonl.gz','rt') as source:row=json.loads(source.readline())
            self.assertEqual((row['step'],row['physical_ids']),(4444,['b','a']))

    def test_repeated_exposure_and_physical_id_alignment(self):
        batch=(None,torch.tensor([-1,-1]),None,{'base_index':torch.tensor([7,4])})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'observations.gz'
            with PseudoObservationWriter(path,dataset()) as writer:
                for step in (1,2):
                    writer.record(step=step,batch=batch,pseudo=torch.tensor([1,0]),
                                  confidence=torch.tensor([.97,.90]),accepted=torch.tensor([True,False]))
                self.assertEqual((writer.calls,writer.seen,writer.accepted),(2,4,2))
            with gzip.open(path,'rt',encoding='utf-8') as source:rows=[json.loads(line) for line in source]
            self.assertEqual(rows[0]['physical_ids'],['b','a'])
            self.assertEqual(rows[1]['pseudo'],[1,0])
            self.assertNotIn('truth',rows[0])

    def test_reject_truth_and_unknown_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            data=dataset();data.metadata['y']=np.array([0,1])
            with self.assertRaises(ValueError):PseudoObservationWriter(Path(folder)/'bad.gz',data)
            with PseudoObservationWriter(Path(folder)/'safe.gz',dataset()) as writer:
                batch=(None,torch.tensor([0]),None,{'base_index':torch.tensor([4])})
                args=dict(step=1,batch=batch,pseudo=torch.tensor([0]),confidence=torch.tensor([.99]),accepted=torch.tensor([True]))
                with self.assertRaises(ValueError):writer.record(**args)
                args['batch']=(None,torch.tensor([-1]),None,{'base_index':torch.tensor([99])})
                with self.assertRaises(ValueError):writer.record(**args)


if __name__=='__main__':unittest.main()
