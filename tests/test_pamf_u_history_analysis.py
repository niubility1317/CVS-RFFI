import gzip
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.pamf_u_history.analyze import records,step_summary,paired_rows


class AnalysisTests(unittest.TestCase):
    def test_live_truncated_gzip_parses_all_complete_records(self):
        rows=[dict(step=i,metrics={'train/loss':float(i)}) for i in range(1,101)]
        payload=''.join(json.dumps(r)+'\n' for r in rows).encode()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'steps.gz';p.write_bytes(gzip.compress(payload)[:-8])
            actual,closed=records(p,complete=False)
            self.assertEqual(actual,rows);self.assertFalse(closed)
            with self.assertRaises(ValueError):records(p,complete=True)

    def test_missing_step_is_rejected(self):
        with self.assertRaises(ValueError):step_summary([dict(step=1),dict(step=3)],20)

    def test_paired_net_fixes_and_harms(self):
        y=np.array([0,1,2]);candidate=np.eye(6)[[0,0,2]][None,:,:];control=np.eye(6)[[1,1,2]][None,:,:]
        r=paired_rows(candidate,control,y,['raw'],'test')[0]
        self.assertEqual((r['wrong_to_right'],r['right_to_wrong'],r['net_fix']),(1,1,0))
        self.assertEqual(r['delta_pp'],0.)


if __name__=='__main__':unittest.main()
