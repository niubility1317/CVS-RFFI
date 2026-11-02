import unittest
import numpy as np
from audit_sample_source import verify_routes

class AuditRouteTest(unittest.TestCase):
 def test_valid_routes_and_tampered_selection(self):
  p=np.array([[[.9,.1],[.1,.9]],[[.9,.1],[.2,.8]]]);lw=np.log(np.array([[.8,.2],[.3,.7]]));choices=np.array([[0,0],[1,0]]);pred=np.array([[0,0,0,0],[0,0,1,0]]);verify_routes(p,lw,choices,pred);bad=choices.copy();bad[1,0]=0
  with self.assertRaises(ValueError):verify_routes(p,lw,bad,pred)
if __name__=='__main__':unittest.main()
