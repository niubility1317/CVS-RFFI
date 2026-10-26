import unittest
import numpy as np
from experiments.pamf_u_history.analyze_covariance import source_rule,paired,VIEWS


class CovarianceAnalysisTests(unittest.TestCase):
    def test_same_packet_alignment_and_fix_harm(self):
        y=np.array([0,1,2]);metadata=dict(ids=np.array(['a','b','c']),y=y,views=np.array(VIEWS),rx=np.array([1,3,4]),day=np.array([1,1,2]))
        a=dict(metadata,logits=np.tile(np.eye(6)[[0,0,2]],(8,1,1)));b=dict(metadata,logits=np.tile(np.eye(6)[[1,1,2]],(8,1,1)))
        rows,agreement=paired(a,b,'CovSel-U');r=rows[0]
        self.assertEqual((r['wrong_to_right'],r['right_to_wrong'],r['net_fix']),(1,1,0))
        self.assertIn('matched SIG1',r['comparison']);self.assertTrue(agreement)
        bad=dict(a,ids=np.array(['b','a','c']))
        with self.assertRaises(ValueError):paired(bad,b,'CovSel-U')
        bad=dict(a,logits=a['logits'].copy());bad['logits'][0,0,0]=np.nan
        with self.assertRaises(ValueError):paired(bad,b,'CovSel-U')

    def test_source_boundary_and_full_U_requirement(self):
        state=dict(calls=6660,seen=1701000,unique_seen=56700,min_visits=30,max_visits=30,covariance_U_seen=1701000,U_truth_read=False,target_reads=0,model_EMA_updates=0)
        rows=[dict(view=v,n=27000,candidate_correct=24000,control_correct=24000,net_fix=0) for v in VIEWS]
        rows[0].update(candidate_correct=23973,net_fix=-27)
        self.assertTrue(source_rule(state,rows)['passed'])
        rows[0].update(candidate_correct=23972,net_fix=-28)
        self.assertFalse(source_rule(state,rows)['passed'])
        rows[0].update(candidate_correct=23973,net_fix=-27)
        self.assertFalse(source_rule(dict(state,covariance_U_seen=1690000),rows)['passed'])
        self.assertTrue(source_rule(state,rows[:-1])['incomplete'])
        rows[0]['n']=26999
        with self.assertRaises(ValueError):source_rule(state,rows)


if __name__=='__main__':unittest.main()
