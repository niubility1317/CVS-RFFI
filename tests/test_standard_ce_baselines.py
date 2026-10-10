import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.standard_ce_baselines.common import ROOT,read,write
from experiments.standard_ce_baselines.source import validate_config
from experiments.standard_ce_baselines.dispatch import validate_spec
from experiments.standard_ce_baselines.predict import check
from experiments.standard_ce_baselines.score import score


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config=read(ROOT/'experiments/standard_ce_baselines/configs/resnet18-s2026092701.json')

    def test_training_rejects_target_checkpoint_and_extra_objectives(self):
        validate_config(self.config)
        for key,value in [('target_truth','hidden.json'),('target_inputs','query.npy'),('checkpoint','old.pt'),
                          ('resume','old.pt'),('augmentation',True),('extra_losses',['contrastive']),('selection','best_target')]:
            changed=dict(self.config);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_config(changed)

    def test_matrix_rejects_duplicates_and_path_escape(self):
        original=read(ROOT/'experiments/standard_ce_baselines/configs/launch_spec.json');validate_spec(original)
        changed=copy.deepcopy(original);changed['rows'][-1]=changed['rows'][0]
        with self.assertRaises(ValueError):validate_spec(changed)
        changed=copy.deepcopy(original);changed['rows'][0]['prediction_output']='/outside/run'
        with self.assertRaises(ValueError):validate_spec(changed)

    def test_checkpoint_freeze_and_provenance(self):
        config=dict(variant='resnet18',model_seed=2026092701,source_output='/source')
        done=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False)
        initial=dict(scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],target_access=False,model_seed=2026092701)
        contract=dict(out_len=256,equalized=1,normalize=True)
        resolved=dict(self.config,source_counts={'L_s':6300,'U_s':56700,'V':27000})
        payload=dict(method='standard_ce_baselines',variant='resnet18',epoch=200,selection='fixed_last_epoch',config=resolved,initialization=initial,source_contract=contract)
        frozen=dict(status='FROZEN',epoch=200,variant='resnet18',model_seed=2026092701,checkpoint=str(Path('/source')/'last.pt'),target_access=False,target_score_used=False)
        check(config,done,initial,contract,contract,resolved,payload,frozen)
        for changed in [dict(frozen,target_score_used=True),dict(frozen,epoch=199)]:
            with self.assertRaises(ValueError):check(config,done,initial,contract,contract,resolved,payload,changed)
        with self.assertRaises(ValueError):check(config,done,dict(initial,ancestors=['old']),contract,contract,resolved,payload,frozen)

    def test_incomplete_prediction_never_opens_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);capsule=root/'capsule';prediction=root/'prediction';capsule.mkdir();prediction.mkdir()
            np.savez(capsule/'index.npz',ids=np.asarray(['opaque-a','opaque-b']))
            write(prediction/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=2,views=['clean'],truth_read=False,query_fit=False))
            np.savez(prediction/'predictions.npz',ids=np.asarray(['opaque-b','opaque-a']),clean=np.asarray([0,1]))
            truth=root/'truth.json'
            from experiments.standard_ce_baselines import score as scoring
            original_read=scoring.read
            calls=[]
            def guarded(path):
                calls.append(str(path))
                if str(path)==str(truth):raise AssertionError('Truth opened before coverage validation')
                return original_read(path)
            with patch.object(scoring,'read',side_effect=guarded),self.assertRaises(ValueError):
                score(dict(prediction_output=str(prediction),capsule=str(capsule),truth=str(truth)))
            self.assertNotIn(str(truth),calls)

if __name__=='__main__':unittest.main()
