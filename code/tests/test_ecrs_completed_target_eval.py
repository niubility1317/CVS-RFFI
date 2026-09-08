import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/eval_ecrs_completed_target_20260908.py'
spec = importlib.util.spec_from_file_location('frozen_target_eval', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_truth_blind_wrapper_strips_label_and_tx_metadata():
    class Data:
        def __len__(self): return 1
        def __getitem__(self, i):
            return torch.zeros(2, 256), object(), 99, {'physical_sample_id': 'sample:0',
                'receiver_id': 2, 'day_id': 0, 'tx_i': 5, 'tx': 'secret', 'label': 5}
    x, y, d, meta = mod.TruthBlindTarget(Data())[0]
    assert y == -1
    assert set(meta) == {'physical_sample_id', 'receiver_id', 'day_id'}


def test_independent_scorer_requires_complete_ids_and_rejects_truth_leak(tmp_path):
    truth = tmp_path/'truth.jsonl'; pred = tmp_path/'pred.jsonl'
    t = [{'physical_sample_id': f'sample:{i}', 'truth': i, 'receiver_id': 2, 'day_id': 0} for i in range(2)]
    p = [{'physical_sample_id': f'sample:{i}', 'receiver_id': 2, 'day_id': 0,
          'predictions': {'raw': 0, 'fused': i}} for i in range(2)]
    truth.write_text(''.join(json.dumps(x)+'\n' for x in t))
    pred.write_text(''.join(json.dumps(x)+'\n' for x in p))
    result = mod.score_files(truth, pred, 2)
    assert result['paths']['raw']['accuracy'] == 50
    assert result['paths']['fused']['accuracy'] == 100
    assert result['rescue'] == 1 and result['harm'] == 0
    pred.write_text(json.dumps(p[0])+'\n')
    with pytest.raises(ValueError, match='incomplete'): mod.score_files(truth, pred, 2)
    p[0]['source_truth'] = 0
    pred.write_text(''.join(json.dumps(x)+'\n' for x in p))
    with pytest.raises(ValueError, match='leaked'): mod.score_files(truth, pred, 2)


def test_b0_rebuild_matches_actual_train_dual_forward():
    from model_dual_cvsincnet import build_dual_model
    torch.set_num_threads(1)
    args = dict(num_classes=6, num_domains=15, model_size='M', dataset='wisig', input_len=256,
        model_variant='lite_d', branch_ablation='no_dac', domain_branch_ablation='no_stats',
        use_ecrs=False, use_mixstyle=False)
    model = build_dual_model(**{k:v for k,v in args.items() if k != 'use_mixstyle'}).eval()
    copy, audit = mod.build_model({'args': args, 'model': model.state_dict(), 'feature_schema': 'ADV3B02:z_id:legacy'}, torch.device('cpu'))
    copy.eval()
    x = torch.randn(2, 2, 256)
    with torch.no_grad():
        torch.testing.assert_close(copy.forward_identity(x)['tx_logits'], model.forward_identity(x)['tx_logits'], atol=0, rtol=0)
    assert audit['checkpoint_load_strict']
