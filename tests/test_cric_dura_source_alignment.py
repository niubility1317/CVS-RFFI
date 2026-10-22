import tempfile,unittest
from pathlib import Path
import numpy as np
from experiments.cric_dura_comparison.f_data import SourceDataset

ROOT=Path(__file__).resolve().parents[1]/'local_artifacts/F_ablation_repair_20261022/tests'
ROOT.mkdir(parents=True,exist_ok=True)

def fixture(root,role,stored=('c','a','b'),declared=('a','b','c')):
    values={'a':1.,'b':2.,'c':3.}
    np.save(root/(role+'_x.npy'),np.asarray([np.full((2,256),values[s],np.float32) for s in stored]))
    np.savez(root/(role+'_index.npz'),ids=np.asarray(stored),y=np.asarray([99]*len(stored)))
    meta={k:np.ones(len(declared),np.int64) for k in ('rx_i','day_i','eq_i','sig_i','base_index','domain')}
    if role!='U_s':meta['y']=np.arange(len(declared),dtype=np.int64)
    np.savez(root/(role+'_metadata.npz'),ids=np.asarray(declared),**meta)
    return dict(counts={role:len(declared)},role_ids={role:list(declared)})

class SourceAlignment(unittest.TestCase):
    def test_actual_IQ_is_matched_to_declared_L_and_V_physical_ID_order(self):
        for role in ('L_s','V'):
            with tempfile.TemporaryDirectory(dir=ROOT) as folder:
                root=Path(folder);contract=fixture(root,role);dataset=SourceDataset(root,role,contract)
                self.assertEqual(dataset.x[:,0,0].tolist(),[1.,2.,3.])
                self.assertEqual([dataset[i][1] for i in range(3)],[0,1,2])
    def test_U_IQ_is_aligned_using_only_ID_and_remains_unlabeled(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root=Path(folder);contract=fixture(root,'U_s');dataset=SourceDataset(root,'U_s',contract)
            self.assertEqual(dataset.x[:,0,0].tolist(),[1.,2.,3.])
            self.assertEqual([dataset[i][1] for i in range(3)],[-1,-1,-1])
    def test_missing_duplicate_or_extra_storage_ID_rejected(self):
        for stored in (('a','a','c'),('a','b','z')):
            with tempfile.TemporaryDirectory(dir=ROOT) as folder:
                root=Path(folder);contract=fixture(root,'L_s')
                np.savez(root/'L_s_index.npz',ids=np.asarray(stored))
                with self.assertRaisesRegex(ValueError,'physical ID'):
                    SourceDataset(root,'L_s',contract)
    def test_identity_storage_order_does_not_change_IQ_or_metadata(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root=Path(folder);contract=fixture(root,'L_s',stored=('a','b','c'))
            original=np.load(root/'L_s_x.npy');dataset=SourceDataset(root,'L_s',contract)
            np.testing.assert_array_equal(dataset.x.numpy(),original)

if __name__=='__main__':unittest.main()
