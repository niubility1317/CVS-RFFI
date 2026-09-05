"""Bounded CPU main-loop integration with synthetic source-only observations."""
import json
import math
import sys
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from SSDG import train_ssdg as train
from model_dual_cvsincnet import DualCVSincNetDisentangle


class SourceRows(Dataset):
    def __init__(self, offset):
        self.offset = offset
        self.iq = torch.randn(2, 2, 256, generator=torch.Generator().manual_seed(offset))

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return self.iq[index], index, 0, dict(rx_i=0, day_i=0, eq_i=0,
            sig_i=index + self.offset, base_index=index + self.offset, tx_i=index)


def test_pair_main_masked_one_epoch_checkpoint_and_optimizer(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    labeled = DataLoader(SourceRows(10), batch_size=2)
    unlabeled = DataLoader(train._MUSEUnlabeledDatasetView(SourceRows(20)), batch_size=2)
    val = DataLoader(SourceRows(30), batch_size=2)
    context = dict(train_loader=labeled, unlabeled_loader=unlabeled, val_loader=val,
        source_calibration_loader=val, probe_train_loader=labeled,
        balanced_train_sampler=None, named_test_loaders={}, domain_label_map={0: 0},
        num_domains=1, input_len=256, class_id_to_tx=['0', '1'],
        split_info={'mode': 'synthetic_source_only', 'labeled_size': 2,
                    'unlabeled_size': 2, 'source_val_size': 2})
    monkeypatch.setattr(train, '_build_ssdg_wisig_data', lambda *a: context)
    # Preserve the real first-stage policy; one epoch cannot fit six schedule stages.
    monkeypatch.setattr(train, '_muse_config_from_args', lambda args: train.MUSEConfig())
    initial = {}
    def build_model(args, device):
        model = DualCVSincNetDisentangle(num_classes=2, num_domains=1, input_len=256).to(device)
        initial.update({key: value.detach().clone() for key, value in model.state_dict().items()})
        return model
    monkeypatch.setattr(train, 'build_baseline_model', build_model)
    argv = ['train_ssdg', '--output_dir', str(tmp_path), '--from_scratch', 'true',
        '--device', 'cpu', '--amp', 'false', '--epochs', '1', '--muse_final_epoch', '1',
        '--label_epochs', '1', '--use_muse_ssdg', 'true', '--muse_level', 'M1',
        '--muse_lr_schedule', 'off', '--pair_reform', 'point', '--pair_start_epoch', '1',
        '--pair_pseudo_start_epoch', '1', '--sat_training_mode', 'concat_masked',
        '--sat_cons_start_epoch', '1',
        '--sat_train_scenario', 'leo_clear_weak', '--eval_sat_channel', 'false',
        '--num_workers', '0', '--batch_size', '2', '--muse_unlabeled_batch_size', '2']
    monkeypatch.setattr(sys, 'argv', argv)
    # This deliberately non-formal synthetic row must complete without promotion.
    assert train.main() == 8
    checkpoint = train.load_checkpoint(tmp_path / 'final_ssdg.pth', torch.device('cpu'))
    assert checkpoint['pair_formula'] == 'pair_reform_v3'
    assert checkpoint['optimizer']['state']
    assert any(not torch.equal(value, checkpoint['model'][key]) for key, value in initial.items()
               if value.is_floating_point())
    rows = [json.loads(line) for line in (tmp_path / 'metrics_epoch.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row['train_optimizer_step_applied'] == 1
    assert math.isfinite(row['train_loss'])
    assert row['train_loss_daot_total'] > 0
    assert row['train_loss_daot_unlabeled'] > 0
    assert row['train_loss'] == pytest.approx(row['train_loss_labeled'] + row['train_loss_unlabeled'])
    terminal = json.loads((tmp_path / 'phase1_terminal_status.json').read_text(encoding='utf-8'))
    assert terminal['status'] == 'NON_PROMOTABLE_P0_DISABLED'
    assert not terminal['promotion_ready']
    heldout = json.loads((tmp_path / 'frozen_phase1_heldout_eval.json').read_text(encoding='utf-8'))
    assert heldout['status'] == 'COMPLETE'
    assert heldout['checkpoint_epoch'] == 1
    assert heldout['test']['tx_total'] == 0  # No held-out data or accuracy claim in this test.
