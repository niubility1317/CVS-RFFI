import json
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .models import UncertaintyRestorer
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_action_restore.runtime import load_checkpoint as load_action
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_residual_restore.runtime import load_coarse
from experiments.pamf_restoration.models import complex_iq
from experiments.pamf_restoration.runner import read_json, array_tensor
from experiments.pamf_restoration.data import write_json

ROOT = Path(__file__).resolve().parents[2]


def action_metadata(parent):
    launch, complete = read_json(parent/'launch.json'), read_json(parent/'complete.json')
    training = read_json(parent/'learned_operator/training_complete.json')
    if (parent.name != d.ACTION_RUN or launch['commit'] != d.ACTION_COMMIT or complete['status'] != 'ANALYZED'
            or not complete['source_only'] or complete['target_reads'] != 0 or complete['training_TX_labels_read'] != 0
            or training['epochs'] != 20 or training['updates'] != 4940 or training['coarse_updates'] != 0
            or training['TX_labels_read'] != 0 or training['F_outputs_read'] != 0 or training['target_reads'] != 0
            or training['L_exposures'] != 126000 or training['U_exposures'] != 1134000):
        raise ValueError('Fixed source-only R07 physical-estimator contract required')
    cache = read_json(parent/'cache/complete.json')
    if (cache['status'] != 'ACTION_SOURCE_CACHE_COMPLETE' or cache['epochs'] != 20 or cache['train_count'] != 63000
            or cache['source_V_count'] != 1800 or cache['TX_labels_read'] or cache['target_reads']):
        raise ValueError('Original complete R07 source cache required')
    return dict(coarse_run=d.COARSE_RUN, coarse_commit=d.COARSE_COMMIT, action_run=d.ACTION_RUN,
        action_commit=d.ACTION_COMMIT, data_contract='original7/63/30', training_TX_F_reads=0, target_training=False,
        action_reused_as='Frozen source-trained estimator,not a previously successful restorer')


def frozen_context(args, kind, device):
    if kind == 'direct_control':
        model, _ = load_coarse(args.parent_coarse, device)
        return model, dict(coarse_run=d.COARSE_RUN, coarse_commit=d.COARSE_COMMIT, action_run=None,
            action_commit=None, data_contract='original7/63/30', training_TX_F_reads=0, target_training=False)
    metadata = action_metadata(args.parent_action)
    model, payload = load_action(args.parent_action/'learned_operator/final.pt', device)
    if payload['epoch'] != 20 or payload['updates'] != 4940:
        raise ValueError('Fixed final R07 estimator required')
    return model.requires_grad_(False).eval(), metadata


def calibration_indices(count):
    if count != 63000:
        raise ValueError('Original full source L/U pool required')
    return np.random.default_rng(d.CALIBRATION_SEED).permutation(count)[:8192]


def validate_calibration(value):
    if (value['status'] != 'SOURCE_EXCESS_FROZEN' or value['count'] != 8192 or
            value['seed'] != d.CALIBRATION_SEED or value['source_V_reads'] != 0 or value['TX_labels_read'] != 0 or
            value['F_outputs_read'] != 0 or value['target_reads'] != 0 or value['optimizer_updates'] != 0 or
            value['L_exposures']+value['U_exposures'] != 8192 or
            value['fitted_scalars'] != 1 or not np.isfinite(value['excess_variance']) or value['excess_variance'] < 1e-6):
        raise ValueError('Source-training-only one-scalar calibration required')


@torch.no_grad()
def calibrate(args):
    model, metadata = frozen_context(args, 'global_floor', 'cuda:0')
    cache = args.parent_action/'cache'
    clean = np.load(cache/'train/clean.npy', mmap_mode='r', allow_pickle=False)
    received = np.load(cache/'epochs/01/received.npy', mmap_mode='r', allow_pickle=False)
    roles = np.load(cache/'train/roles.npy', allow_pickle=False)
    index = calibration_indices(len(clean)); residual_sum = native_sum = elements = 0
    for begin in range(0, len(index), d.BATCH_SIZE):
        selected = index[begin:begin+d.BATCH_SIZE]
        x, r = array_tensor(clean[selected], 'cuda:0'), array_tensor(received[selected], 'cuda:0')
        _, action, native = model.estimate(r)
        error = complex_iq(r)-action(complex_iq(x))
        residual_sum += float(error.abs().square().sum())/2
        native_sum += float(native.sum())*error.shape[-1]
        elements += error.numel()
    residual, native = residual_sum/elements, native_sum/elements
    value = dict(status='SOURCE_EXCESS_FROZEN', count=8192, seed=d.CALIBRATION_SEED,
        excess_variance=max(residual-native, 1e-6), real_residual_variance_mean=residual,
        native_real_variance_mean=native, raw_excess=residual-native,
        L_exposures=int((roles[index] == 'L_s').sum()), U_exposures=int((roles[index] == 'U_s').sum()),
        fitted_scalars=1, optimizer_updates=0, source_V_reads=0, TX_labels_read=0, F_outputs_read=0, target_reads=0,
        parent=metadata)
    validate_calibration(value); write_json(args.output/'calibration.json', value)
    print(json.dumps(dict(event='SOURCE_EXCESS_FROZEN', **value)), flush=True)


def save_checkpoint(path, model, optimizer, epoch, updates, parent, calibration=None):
    torch.save(dict(schema='uncertainty_checkpoint_v1', config=d.configuration(), kind=model.kind,
        model=model.state_dict(), optimizer=optimizer.state_dict() if optimizer is not None else None,
        epoch=epoch, updates=updates, parent=parent, calibration=calibration,
        training_TX_labels_read=0, training_F_access=False, target_training=False), path)


def validate_payload(payload):
    kind = payload['kind']; parent = payload['parent']
    if (payload['schema'] != 'uncertainty_checkpoint_v1' or payload['config'] != d.configuration()
            or kind not in (*d.NAMES, 'global_floor') or payload['training_TX_labels_read'] != 0
            or payload['training_F_access'] or payload['target_training']
            or parent['coarse_run'] != d.COARSE_RUN or parent['coarse_commit'] != d.COARSE_COMMIT
            or parent['data_contract'] != 'original7/63/30' or parent['training_TX_F_reads'] != 0 or parent['target_training']):
        raise ValueError('Uncertainty checkpoint provenance/config/permissions differ')
    if kind == 'direct_control':
        if parent['action_run'] is not None or parent['action_commit'] is not None:
            raise ValueError('Direct control must not use an action estimator')
    elif parent['action_run'] != d.ACTION_RUN or parent['action_commit'] != d.ACTION_COMMIT:
        raise ValueError('Fixed source-only action parent required')
    if kind in ('uncertain_map_prior', 'global_floor'):
        validate_calibration(payload['calibration'])
    if kind == 'global_floor':
        if payload['epoch'] != 0 or payload['updates'] != 0:
            raise ValueError('Global-floor control has no optimizer updates')
    elif not 1 <= payload['epoch'] <= 20 or not 0 <= payload['updates'] <= 4940:
        raise ValueError('Registered incremental budget differs')


def load_checkpoint(path, device):
    payload = torch.load(path, map_location=device, weights_only=False); validate_payload(payload)
    kind = payload['kind']
    frozen = ComplexTFUNet() if kind == 'direct_control' else ActionRestorer(ComplexTFUNet())
    floor = payload['calibration']['excess_variance'] if kind in ('uncertain_map_prior', 'global_floor') else None
    model = UncertaintyRestorer(kind, frozen, floor).to(device)
    model.load_state_dict(payload['model'], strict=True)
    if floor is not None and abs(float(model.excess_variance)-floor) > 1e-6*max(1., floor):
        raise ValueError('Stored source excess buffer differs')
    return model.eval(), payload
