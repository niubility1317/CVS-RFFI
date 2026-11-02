import unittest
import numpy as np
from gate_aware_router import GateAwareRouter
from audit_gate_router_ground import verify_block

class GroundRouteAuditTest(unittest.TestCase):
 def test_fixed_model_and_candidate_selection_are_recomputed(self):
  net=GateAwareRouter().eval();c=np.array([[0,1,2,3,4],[3,3,3,3,3]],dtype=np.int64);g=np.array([.9,.2],dtype=np.float32);route=np.array([1,1],dtype=np.uint8);raw=c[:,0].copy();rest=c[np.arange(2),route];self.assertEqual(verify_block(net,c,g,route,raw,rest),2)
 def test_tampered_route_is_rejected(self):
  net=GateAwareRouter().eval();c=np.array([[0,1,2,3,4]],dtype=np.int64);g=np.array([.9],dtype=np.float32);route=np.array([0],dtype=np.uint8)
  with self.assertRaises(ValueError):verify_block(net,c,g,route,c[:,0],c[:,0])
if __name__=='__main__':unittest.main()
