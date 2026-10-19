from pathlib import Path
import numpy as np
import torch
from . import design as d
from .views import AnalyticViews, ViewLoss, scales_from_sums
from experiments.pamf_uncertainty_restore.runtime import load_checkpoint as load_parent, validate_payload as validate_parent
from experiments.pamf_uncertainty_restore.models import UncertaintyRestorer
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_restoration.runner import read_json, array_tensor
from experiments.pamf_restoration.data import write_json

ROOT = Path(__file__).resolve().parents[2]


def parent_metadata(parent):
    if parent.name != d.PARENT or read_json(parent/'launch.json')['commit'] != d.PARENT_COMMIT:
        raise ValueError('Original R10 native source parent required')
    complete = read_json(parent/'complete.json')
    trained = read_json(parent/d.KIND/'training_complete.json')
    expected = dict(status='SOURCE_TRAINED', kind=d.KIND, epochs=20, updates=4940,
        L_exposures=126000, U_exposures=1134000, TX_labels_read=0, F_outputs_read=0, target_reads=0, frozen_context_updates=0)
    if (complete['status'] != 'ANALYZED' or not complete['source_only'] or complete['target_rows']
            or complete['training_TX_F_access'] or any(trained.get(key) != value for key, value in expected.items())):
        raise ValueError('Original zero-TX/F source parent and full inherited budget required')
    return dict(run=d.PARENT, commit=d.PARENT_COMMIT, candidate=d.KIND, source_contract='original7/63/30',
        selection='Original source-only native evidence,not R11 target scores', source_selected=True,
        inherited_prior_epochs=20, inherited_component_epochs=60, training_TX_F_reads=0, target_training=False)


def initialize(parent, device):
    metadata = parent_metadata(parent)
    model, original = load_parent(parent/d.KIND/'final.pt', device)
    if original['kind'] != d.KIND or original['epoch'] != 20 or original['updates'] != 4940:
        raise ValueError('Fixed final R10 native prior required')
    provenance = {key: value for key, value in original.items() if key not in ('model', 'optimizer')}
    model.frozen.requires_grad_(False).eval(); model.prior.requires_grad_(True)
    return model, metadata, provenance


def validate_calibration(value):
    if (value.get('status') != 'ANALYTIC_RMS_FROZEN' or value.get('count') != 8192 or value.get('seed') != d.CALIBRATION_SEED
            or any(value.get(key) != 0 for key in ('source_V_reads', 'TX_labels_read', 'F_access', 'target_reads'))
            or value.get('optimizer_updates') != 0
            or value.get('fitted_scalars') != 63 or value['L_exposures']+value['U_exposures'] != 8192):
        raise ValueError('Training-source-only fixed 8192 LU analytic normalization required')
    ViewLoss(value['scales'])


@torch.no_grad()
def calibrate(args):
    if args.parent_action.name != d.ACTION:
        raise ValueError('Original complete R07 source cache required')
    cache = args.parent_action/'cache'
    complete = read_json(cache/'complete.json')
    if complete['train_count'] != 63000 or complete['TX_labels_read'] or complete['target_reads']:
        raise ValueError('Original legal source cache required')
    clean = np.load(cache/'train/clean.npy', mmap_mode='r', allow_pickle=False)
    roles = np.load(cache/'train/roles.npy', allow_pickle=False)
    if clean.shape != (63000, 2, 256) or (roles == 'L_s').sum() != 6300 or (roles == 'U_s').sum() != 56700:
        raise ValueError('All original legal L/U required')
    selected = np.random.default_rng(d.CALIBRATION_SEED).permutation(63000)[:8192]
    views = AnalyticViews().to('cuda:0'); sums = {}; counts = {}
    for begin in range(0, len(selected), 256):
        parts = views(array_tensor(clean[selected[begin:begin+256]], 'cuda:0'))
        for key, value in parts.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite source analytic target')
            squared = value.double().square().sum((0, *range(2, value.ndim))).cpu()
            sums[key] = sums.get(key, torch.zeros_like(squared))+squared
            counts[key] = counts.get(key, 0)+len(value)*(value.shape[-1] if value.ndim == 3 else 1)
    scales = scales_from_sums(sums, counts)
    result = dict(status='ANALYTIC_RMS_FROZEN', count=8192, seed=d.CALIBRATION_SEED,
        source_V_reads=0, TX_labels_read=0, F_access=0, target_reads=0, fitted_scalars=63, optimizer_updates=0,
        L_exposures=int((roles[selected]=='L_s').sum()), U_exposures=int((roles[selected]=='U_s').sum()),
        scales={key: value.tolist() for key, value in scales.items()})
    validate_calibration(result); write_json(args.output/'calibration.json', result)


def save_checkpoint(path, model, optimizer, epoch, updates, row, parent, provenance, calibration):
    torch.save(dict(schema='analytic_restore_checkpoint_v1', config=d.configuration(), kind=d.KIND, candidate=row,
        model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, updates=updates,
        parent=parent, parent_payload=provenance, calibration=calibration,
        training_TX_labels_read=0, training_F_access=False, target_training=False), path)


def validate_payload(payload):
    if (payload.get('schema') != 'analytic_restore_checkpoint_v1' or payload.get('config') != d.configuration()
            or payload.get('kind') != d.KIND or payload.get('candidate') not in d.ROWS
            or payload.get('training_TX_labels_read') != 0 or payload.get('training_F_access') is not False
            or payload.get('target_training') is not False or not 1 <= payload.get('epoch', 0) <= 20
            or not 1 <= payload.get('updates', 0) <= 4940):
        raise ValueError('Analytic continuation contract/permissions differ')
    parent = payload['parent']
    if (parent['run'] != d.PARENT or parent['commit'] != d.PARENT_COMMIT or parent['candidate'] != d.KIND
            or parent['source_contract'] != 'original7/63/30' or parent['training_TX_F_reads'] != 0
            or parent['target_training'] or parent['inherited_prior_epochs'] != 20):
        raise ValueError('Fixed source-only full native ancestry required')
    validate_parent(payload['parent_payload'])
    if payload['parent_payload']['kind'] != d.KIND or payload['parent_payload']['epoch'] != 20 or payload['parent_payload']['updates'] != 4940:
        raise ValueError('Original last20/4940 source parent required')
    if payload['candidate'] == 'analytic_views':
        validate_calibration(payload['calibration'])
    elif payload['calibration'] is not None:
        raise ValueError('Control must exclude unused analytic training branch')


def load_checkpoint(path, device):
    payload = torch.load(path, map_location=device, weights_only=False); validate_payload(payload)
    model = UncertaintyRestorer(d.KIND, ActionRestorer(ComplexTFUNet())).to(device)
    model.load_state_dict(payload['model'], strict=True)
    return model.eval(), payload
