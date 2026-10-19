"""Inference-only exports of two existing, provenance-checked PAMF models.

AsF exports exactly its identity encoder and original cosine head. Its unused
domain/teacher/training branches are not inference inputs. Centered Sinc
coordinates are evaluated as origin+25000*coordinate, then frozen in Hz; no
weight fitting, interpolation, partial provenance bypass or new head occurs.
"""
import math
from pathlib import Path
import json
import torch
from torch import nn
from experiments.standard_ce_baselines.sigtime import SIGTimeCVS, path_signature


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class SignatureView(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local_projection = nn.Conv1d(channels, 6, 1, bias=False)
        self.register_buffer('omega', torch.zeros(14, 6))
        self.register_buffer('descriptor_lift', torch.eye(28))

    def forward(self, h):
        u = torch.tanh(self.local_projection(h)).transpose(1, 2)
        batch, length, _ = u.shape
        windows = u.reshape(batch*4, length//4, 6)
        return (path_signature(windows)@self.descriptor_lift).reshape(batch, 112)


class TripleSignaturePool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.views = nn.ModuleList(SignatureView(channels) for _ in range(3))
        self.output_projection = nn.Linear(336, 160, bias=False)
        self.output_norm = nn.LayerNorm(160)
        self.gain_raw = nn.Parameter(torch.full((160,), math.atanh(.25)))

    def forward(self, h):
        descriptor = torch.cat([view(h) for view in self.views], 1)
        return .2*self.gain_raw.tanh()*self.output_norm(self.output_projection(descriptor))


def check_contract(folder, source_cache):
    actual = read(folder/'source_contract.json')
    expected = read(Path(source_cache).parent/'source/source_contract.json')
    for field in ('role_ids', 'source_rxs', 'source_days'):
        if actual.get(field) != expected.get(field):
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: '+field)
    if 'classes' in actual and actual['classes'] != read(Path(source_cache)/'manifest.json')['class_id_to_tx']:
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: class mapping')
    return actual


def load_readers(bundle, source_cache, device):
    bundle = Path(bundle)
    readers = {}
    clean = bundle/'cleanF'
    contract = check_contract(clean, source_cache)
    initial = read(clean/'initialization.json')
    done = read(clean/'completion.json')
    freeze = read(clean/'freeze.json')
    payload = torch.load(clean/'last.pt', map_location='cpu', weights_only=False)
    if (initial.get('scratch_only') is not True or initial.get('checkpoint_sources') or initial.get('ancestors') or
            initial.get('target_access') is not False or done['target_access'] is not False or
            done['epoch'] != 200 or done['steps'] != 10000 or freeze['status'] != 'FROZEN' or
            payload['variant'] != 'sig_time' or payload['selection'] != 'fixed_last_epoch' or
            payload['source_contract'] != contract or payload['initialization'] != initial):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: cleanF')
    model = SIGTimeCVS(2026092701)
    model.load_state_dict(payload['model'], strict=True)
    readers['cleanF'] = model.to(device).eval().requires_grad_(False)
    as_folder = bundle/'AsF'
    parent = bundle/'AsF_parent'
    check_contract(as_folder, source_cache)
    check_contract(parent, source_cache)
    initial = read(as_folder/'initialization.json')
    ancestor = read(parent/'initialization.json')
    done = read(as_folder/'completion.json')
    parent_done = read(parent/'completion.json')
    freeze = read(as_folder/'source_frozen.json')
    if (ancestor.get('scratch_only') is not True or ancestor.get('checkpoint_sources') or ancestor.get('ancestors') or
            ancestor.get('target_contact') is not False or ancestor.get('source_roles') != 'EXACT_MATCH' or
            initial.get('source_roles') != 'EXACT_MATCH' or initial.get('target_contact') is not False or
            len(initial.get('checkpoint_sources', [])) != 1 or not initial['checkpoint_sources'][0].endswith('/stage20/final_ssdg.pth') or
            done.get('target_access') is not False or parent_done.get('target_access') is not False or
            done.get('head_fit_updates') != 0 or parent_done.get('head_fit_updates') != 0 or
            done['config'] != freeze['config'] or freeze['selection'] != 'final_only'):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: AsF lineage')
    # baseline_origin_sat_view.SatViewStage is the original trusted metadata
    # dataclass, resolved from this checkout's code path by the caller.
    payload = torch.load(as_folder/'final_ssdg.pth', map_location='cpu', weights_only=False)
    if (payload['args']['teacher_ckpt'] or payload['args']['baseline_ckpt'] != initial['checkpoint_sources'][0] or
            payload['epoch'] != 10 or payload['checkpoint_selection'] != 'final_only' or
            payload['run_id'] != done['config']['run_id'] or payload['candidate_id'] != done['config']['row_id']):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: AsF final state')
    prefix = 'id_backbone.encoder.'
    state = {k[len(prefix):]: v for k, v in payload['model'].items() if k.startswith(prefix)}
    sinc_prefix = 'core.id_backbone.sinc.'
    for short, native in (('low', 'low_hz_'), ('band', 'band_hz_')):
        state[sinc_prefix+native] = state.pop(sinc_prefix+short+'_origin_hz')+25000.*state.pop(sinc_prefix+short+'_coordinate')
    model = SIGTimeCVS(2026092701)
    model.sequence_pool = TripleSignaturePool(model.core.id_backbone.t2.pw.out_channels)
    model.load_state_dict(state, strict=True)
    readers['AsF'] = model.to(device).eval().requires_grad_(False)
    for reader in readers.values():
        if reader.core.id_backbone.cls_head.weight.shape != (6, 160):
            raise ValueError('Original default head shape differs')
    return readers


def classification_statistics(labels, prediction, baseline):
    truth = torch.tensor(labels.tolist() if hasattr(labels, 'tolist') else labels, dtype=torch.long)
    pred = torch.tensor(prediction.tolist() if hasattr(prediction, 'tolist') else prediction, dtype=torch.long)
    before = torch.tensor(baseline.tolist() if hasattr(baseline, 'tolist') else baseline, dtype=torch.long)
    confusion = torch.bincount(truth*6+pred, minlength=36).reshape(6, 6).double()
    precision = confusion.diagonal()/confusion.sum(0).clamp_min(1)
    recall = confusion.diagonal()/confusion.sum(1).clamp_min(1)
    f1 = 2*precision*recall/(precision+recall).clamp_min(1e-12)
    correct, old_correct = pred == truth, before == truth
    return dict(accuracy=float(correct.double().mean()*100), macro_f1=float(f1.mean()*100),
                baseline_accuracy=float(old_correct.double().mean()*100),
                repaired=int((correct & ~old_correct).sum()), damaged=int((~correct & old_correct).sum()),
                net_repaired=int(correct.sum()-old_correct.sum()), count=len(truth),
                confusion=confusion.long().tolist())
