import unittest
import numpy as np
import torch
from experiments.phase1_blind_polyphase.source_equivalence import compare_pair,positions_for_source
from experiments.pamf_phase1_flow import experiment as core

class SourceEquivalenceTest(unittest.TestCase):
    def test_mismatch_counts_and_error(self):
        a=torch.zeros(2,2,256);b=a.clone();b[1,0,0]=.01
        x=torch.tensor([[2.,1.,0.,0.,0.,0.],[2.,1.,0.,0.,0.,0.]])
        y=x.clone();y[1,1]=3.
        r=compare_pair(a,b,x,y)
        self.assertEqual(r['decision_mismatches'],1)
        self.assertAlmostEqual(r['iq_max_abs'],.01,places=6)
        self.assertEqual(r['logits_max_abs'],2.)
    def test_source_cell_coverage(self):
        rx=[];scene=[]
        for r in (4,8):
            for v in core.VIEWS:
                rx.extend([r]*8);scene.extend([v]*8)
        self.assertEqual(len(positions_for_source(np.array(rx),np.array(scene))),112)
        with self.assertRaises(ValueError):positions_for_source(np.array(rx[:-1]),np.array(scene[:-1]))

if __name__=='__main__':unittest.main()
