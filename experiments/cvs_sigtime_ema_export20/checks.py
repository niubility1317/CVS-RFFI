import copy,inspect,os,tempfile,unittest
from contextlib import contextmanager,nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from . import design as d
from . import protocol,evaluate

class ExportProtocol(unittest.TestCase):
    def metrics(self,errors=400,CE=.1,count=27000):return dict(count=count,errors=errors,mean_CE=CE)
    def test_source_builder_receives_required_device_without_data_access(self):
        import torch
        from experiments.cvs_phase1_stack import runtime
        from experiments.cvs_equivariant_identity import precision
        with runtime.installed(d.parent_config(2026092701)) as native:
            signature=inspect.signature(native._build_ssdg_wisig_data)
        class ReachedBuilder(Exception):pass
        def bind_without_reading(*args,**kwargs):
            bound=signature.bind(*args,**kwargs)
            self.assertEqual(bound.arguments['device'],torch.device('cuda:0'))
            raise ReachedBuilder
        @contextmanager
        def installed_without_data(config):
            yield SimpleNamespace(_build_ssdg_wisig_data=bind_without_reading)
        with tempfile.TemporaryDirectory() as root,patch.object(d,'BASE',Path(root)),patch.object(d,'read',return_value={}),patch.object(d.parent,'make_args',return_value=SimpleNamespace()),patch.object(evaluate,'parent_source',return_value=Path('own-parent')),patch.object(runtime,'installed',installed_without_data),patch.object(precision,'numerical_context',return_value=nullcontext()),patch('torch.load') as load:
            with self.assertRaises(ReachedBuilder):evaluate.source_compare(d.config(d.rows()[0]))
            load.assert_not_called()
    def test_complete_positive_gate(self):
        self.assertTrue(d.qualify(self.metrics(),self.metrics(399,.09)))
        self.assertTrue(d.qualify(self.metrics(),self.metrics(400,.09)))
    def test_no_improvement_or_harm_does_not_qualify(self):
        self.assertFalse(d.qualify(self.metrics(),self.metrics()))
        self.assertFalse(d.qualify(self.metrics(),self.metrics(401,.09)))
        self.assertFalse(d.qualify(self.metrics(),self.metrics(399,.11)))
    def test_incomplete_nonfinite_source_rejected(self):
        with self.assertRaises(ValueError):d.qualify(self.metrics(),self.metrics(count=100))
        with self.assertRaises(ValueError):d.qualify(self.metrics(),self.metrics(CE=float('nan')))
    def test_fixed_key_seed_and_parent_contract(self):
        for key,value in (('checkpoint_state_key','model'),('model_seed',2026092704),('checkpoint_training_data','four RX')):
            c=copy.deepcopy(d.config(d.rows()[0]));c[key]=value
            with self.assertRaises(ValueError):d.validate(c)
    def test_bad_provenance_rejected_before_checkpoint_load(self):
        with patch.object(protocol.parent_proof,'source_proof',side_effect=ValueError('contract mismatch')),patch('torch.load') as load:
            with self.assertRaises(ValueError):protocol.load_parent(d.config(d.rows()[0]),'cpu',require_frozen=False)
            load.assert_not_called()
    def test_incomplete_ema_rejected_before_model_construction(self):
        import torch
        with patch.object(protocol,'parent_source',return_value=Path('own-parent')),patch.object(protocol.parent_proof,'validate_checkpoint'),patch('torch.load',return_value=dict(model={'W':torch.ones(2)},ema_model={})),patch.object(protocol.parent_model,'build') as build:
            with self.assertRaises(ValueError):protocol.load_parent(d.config(d.rows()[0]),'cpu',require_frozen=False)
            build.assert_not_called()
    def test_frozen_export_required(self):
        with patch.object(protocol,'parent_source',return_value=Path('own-parent')),patch.object(d,'read',side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):protocol.frozen_proof(d.config(d.rows()[0]))
    def test_incomplete_prediction_blocks_truth(self):
        from experiments.cvs_report_ssl import evaluate as truth_side
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=''),patch.object(evaluate,'preflight',side_effect=ValueError('incomplete')),patch.object(truth_side,'truth_arrays') as truth:
            with self.assertRaises(ValueError):evaluate.score(d.rows()[0]['row_id'])
            truth.assert_not_called()
    def test_GPU_scorer_rejected_before_preflight(self):
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'),patch.object(evaluate,'preflight') as preflight:
            with self.assertRaises(ValueError):evaluate.score(d.rows()[0]['row_id'])
            preflight.assert_not_called()
    def test_GPU_aggregation_rejected_before_preflight(self):
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'),patch.object(evaluate,'preflight') as preflight:
            with self.assertRaises(ValueError):evaluate.finish()
            preflight.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
