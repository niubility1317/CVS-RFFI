"""Frozen full-source V sensitivity; received transforms are never training views.

No target input, optimization, checkpoint writing or candidate selection occurs.
These are interventions on already received/equalized IQ, not identifiable TX
hardware changes and not a reconstruction of the upstream WiSig equalizer.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'
BASE = PROJECT + '/releases/cvs_equivariant_fp32_identity_20261002_r01'
BASE_COMMIT = 'be0aec8f2e2a3b50ce41d8ec1778c28c96e7eb6e'
RUN = '20261002-diagnostic-cvs-received-sensitivity-source-manysig-m56-r01'
RELEASE = 'cvs_received_sensitivity_source_20261002_r01'
SEEDS = (2026092701, 2026092702, 2026092703, 2026092704)
SOURCE_RUNS = dict(residual_fusion='20261001-phase1-cvs-residual-identity-manysig-m8-r01',
                   equivariant_memory='20261002-phase1-cvs-equivariant-fp32-manysig-m4-r01')
TRANSFORMS = ('identity', 'constant_phase', 'cfo_plus', 'cfo_minus',
              'received_lti', 'received_image', 'received_cubic')
STAGES = ('id_backbone.t_proj', 'id_backbone.f_proj', 'id_backbone.pa_proj',
          'id_backbone.cls_head.base_norm', 'id_backbone.cls_head.pa_norm')
PARAMETERS = dict(sample_rate_hz=25000000, phase_radians=.37,
                  cfo_hz=80000., lti_delay_samples=1, lti_tap=[.15, .08],
                  image_coefficient=[.02, .02], cubic_coefficient=[-.03, 0.],
                  lti_left_boundary='zero history; first sample does not wrap',
                  rms_after_transform=True)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def transform(x, name):
    import torch
    if name not in TRANSFORMS or x.ndim != 3 or x.shape[1:] != (2, 256):
        raise ValueError('Unregistered received transform or input shape')
    if name == 'identity':
        return x.clone()
    r, i = x.unbind(1)
    if name in ('constant_phase', 'cfo_plus', 'cfo_minus'):
        phase = x.new_tensor(PARAMETERS['phase_radians']) if name == 'constant_phase' else (
            (1 if name == 'cfo_plus' else -1) * 2 * math.pi * PARAMETERS['cfo_hz'] /
            PARAMETERS['sample_rate_hz'] * torch.arange(256, device=x.device, dtype=x.dtype))
        c, s = phase.cos(), phase.sin()
        a, b = r*c-i*s, r*s+i*c
    elif name == 'received_lti':
        # Defined finite-record boundary; do not claim periodic history or exact channel simulation.
        dr, di = torch.nn.functional.pad(r[:, :-1], (1, 0)), torch.nn.functional.pad(i[:, :-1], (1, 0))
        tr, ti = PARAMETERS['lti_tap']
        a, b = r+tr*dr-ti*di, i+tr*di+ti*dr
    elif name == 'received_image':
        cr, ci = PARAMETERS['image_coefficient']
        a, b = r+cr*r+ci*i, i+ci*r-cr*i
    else:
        cr, ci = PARAMETERS['cubic_coefficient']
        p = r.square()+i.square()
        a, b = r+(cr*r-ci*i)*p, i+(cr*i+ci*r)*p
    y = torch.stack((a, b), 1)
    y = y / (y.square().sum(1).mean(-1).sqrt().clamp_min(1e-8)[:, None, None])
    if not torch.isfinite(y).all():
        raise ValueError('Nonfinite received diagnostic view')
    return y


def capture(model, x):
    """Observation hooks return None; every classifier path still executes."""
    import torch
    result, handles = {}, []
    modules = dict(model.named_modules())
    def observe(name):
        def hook(module, inputs, output):
            if name in result:
                raise ValueError('Unexpected repeated diagnostic stage')
            result[name] = output.detach().clone()
        return hook
    try:
        for name in STAGES:
            handles.append(modules[name].register_forward_hook(observe(name)))
        with torch.no_grad():
            result['features'] = model.features(x).detach()
            result['logits'] = model.id_backbone.cls_head.classify(result['features']).detach()
        if set(result) != set(STAGES) | {'features', 'logits'}:
            raise ValueError('Missing executed identity diagnostic stage')
        if any(not torch.isfinite(v).all() for v in result.values()):
            raise ValueError('Nonfinite source observations')
        return result
    finally:
        for handle in handles:
            handle.remove()


def observations(reference, changed, labels):
    import torch
    from torch.nn import functional as F
    pred, original = changed['logits'].argmax(1), reference['logits'].argmax(1)
    metrics = dict(correct=(pred == labels).double(), reference_correct=(original == labels).double(),
                   agreement=(pred == original).double(),
                   logit_max_abs_error=(changed['logits']-reference['logits']).abs().max(1).values.double())
    for name in ('features', *STAGES):
        a, b = [F.normalize(v[name].flatten(1), dim=1, eps=1e-4) for v in (reference, changed)]
        metrics[name+'_unit_distance'] = (a-b).norm(dim=1).double()
    for name, v in metrics.items():
        if len(v) != len(labels) or not torch.isfinite(v).all():
            raise ValueError('Invalid per-packet source diagnostic metric: '+name)
    return metrics


class Groups:
    def __init__(self):
        self.values = {}

    def add(self, batch, metrics):
        import torch
        keys = list(zip(batch['label'].tolist(), batch['receiver'].tolist(), batch['day'].tolist()))
        local = {}
        for j, key in enumerate(keys):
            local.setdefault(key, []).append(j)
        cpu = {k: v.detach().cpu() for k, v in metrics.items()}
        for key, ids in local.items():
            g = self.values.setdefault(key, dict(count=0, sums={}, maxima={}))
            g['count'] += len(ids)
            index = torch.tensor(ids, dtype=torch.long)
            for name, v in cpu.items():
                selected = v[index]
                g['sums'][name] = g['sums'].get(name, 0.) + float(selected.sum())
                g['maxima'][name] = max(g['maxima'].get(name, 0.), float(selected.max()))

    def result(self, expected_count=None):
        count = sum(g['count'] for g in self.values.values())
        if not count or (expected_count is not None and count != expected_count):
            raise ValueError('Incomplete full-source diagnostic count')
        names = next(iter(self.values.values()))['sums']
        summary = dict(count=count,
            mean={k: math.fsum(g['sums'][k] for g in self.values.values())/count for k in names},
            maximum={k: max(g['maxima'][k] for g in self.values.values()) for k in names})
        summary['groups'] = [dict(tx=tx, receiver=rx, day=day, count=g['count'],
            mean={k: v/g['count'] for k, v in g['sums'].items()}, maximum=g['maxima'])
            for (tx, rx, day), g in sorted(self.values.items())]
        return summary


def verify_frozen(row, expected):
    import torch
    from experiments.cvs_equivariant_identity.dispatch import read_source_record
    folder = Path(row['source_output'])
    method = 'cvs_equivariant_identity' if row['variant'] == 'equivariant_memory' else 'cvs_residual_identity'
    record = read_source_record(row, expected, method)
    done, initial, contract, resolved = [read(folder/n) for n in
        ('completion.json', 'initialization.json', 'source_contract.json', 'resolved_config.json')]
    payload = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
    # The immutable SOURCE release has no clean-eval dependency closure. Reuse
    # its complete source provenance verifier and check actual checkpoint payload
    # against that verified provenance, without importing any target evaluator.
    actual = dict(epoch=200, source_contract=contract, initialization=initial,
        selection='fixed_last_epoch', method=method, variant=row['variant'],
        config=resolved, classes=contract['classes'], num_classes=6)
    if any(payload.get(k) != v for k, v in actual.items()):
        raise ValueError('Frozen checkpoint payload disagrees with verified source provenance')
    return payload, resolved, record


def main(output, device):
    if (Path(BASE)/'release_commit.txt').read_text().strip() != BASE_COMMIT:
        raise ValueError('Immutable source implementation mismatch')
    sys.path[:0] = [BASE, BASE+'/code']
    import torch
    from baselines.common.practical_source import build_contract_split
    from baselines.common.cvs_data import make_cvs_loader
    from experiments.cvs_identity_ce.source import source_args
    from experiments.cvs_residual_identity.model import build as residual_build
    from experiments.cvs_equivariant_identity.model import build as equivariant_build
    from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    initial_flags = actual_flags()
    expected = read(PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
    args = source_args(dict(source_contract=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json',
                            dataset=PROJECT+'/Dataset_WigSig/ManySig.pkl', output_root=str(output/'data_contract')))
    split = build_contract_split(args)
    if split.test or split.named_tests or split.split_info['counts'] != dict(L_s=6300, U_s=56700, V=27000):
        raise ValueError('Source roles changed or targets constructed')
    loader = make_cvs_loader(split.val, batch_size=256, shuffle=False, num_workers=0, device=device, drop_last=False)
    write(output/'resolved_config.json', dict(run_id=RUN, pid=os.getpid(), python=sys.executable, cwd=os.getcwd(), argv=sys.argv,
        commit=(Path(__file__).parent/'release_commit.txt').read_text().strip(), model_code_commit=BASE_COMMIT,
        torch_version=torch.__version__, hardware=torch.cuda.get_device_name(device), initial_flags=initial_flags,
        source_counts=split.split_info['counts'], accessed_role='V', count=27000, optimizer_steps=0,
        source_runs=SOURCE_RUNS, seeds=SEEDS, transforms=TRANSFORMS, parameters=PARAMETERS, observation_stages=STAGES,
        target_access=False, target_truth_access=False, target_score_used=False, model_updated=False,
        transform_scope='After existing equalization/crop/RMS; received interventions, not recovered TX changes',
        stage_scope='t/f/pa projections before optional stats; base_norm/pa_norm include full existing classifier paths',
        candidate_selection=False, training_augmentation=False))
    completed = []
    for variant, source_run in SOURCE_RUNS.items():
        for seed in SEEDS:
            start = time.perf_counter()
            row = dict(variant=variant, model_seed=seed, source_output=PROJECT+'/runs/'+source_run+'/'+variant+'-s'+str(seed)+'/source')
            payload, resolved, provenance = verify_frozen(row, expected)
            model = (equivariant_build if variant == 'equivariant_memory' else residual_build)(variant)
            model.to(device).eval().requires_grad_(False)
            model.load_state_dict(payload['model'], strict=True)
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            sums = {name: Groups() for name in TRANSFORMS}
            with numerical_context(resolved.get('numerical_policy')):
                effective_flags = actual_flags()
                with torch.no_grad():
                    smoke = model(torch.zeros(2, 2, 256, device=device))
                    if smoke.shape != (2, 6) or not torch.isfinite(smoke).all():
                        raise ValueError('Real frozen no-query checkpoint smoke failed')
                    for n, batch in enumerate(loader, 1):
                        x, labels = batch['iq'].to(device), batch['label'].to(device)
                        reference = capture(model, x)
                        for name in TRANSFORMS:
                            changed = capture(model, transform(x, name))
                            sums[name].add(batch, observations(reference, changed, labels))
                        if n % 20 == 0:
                            print('SOURCE_PROGRESS '+json.dumps(dict(variant=variant, seed=seed, batches=n, seconds=time.perf_counter()-start)), flush=True)
                results = {name: groups.result(27000) for name, groups in sums.items()}
                if any(len(v['groups']) != 90 or any(g['count'] != 300 for g in v['groups']) for v in results.values()):
                    raise ValueError('Full 6TX x5RX x3day source V coverage differs')
                if abs(results['identity']['mean']['correct']-provenance['accuracy']) > 1e-12:
                    raise ValueError('Unchanged source identity inference did not reproduce fixed E200 accuracy')
                if any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()):
                    raise ValueError('Frozen source model state changed')
                for name, result in results.items():
                    rid = variant+'-s'+str(seed)+'-'+name
                    folder = output/rid; folder.mkdir(exist_ok=False)
                    result.update(row_id=rid, variant=variant, seed=seed, transform=name,
                        checkpoint=row['source_output']+'/last.pt', provenance=provenance, effective_flags=effective_flags,
                        numerical_policy=resolved.get('numerical_policy'), state_unchanged=True, target_access=False,
                        optimizer_steps=0, model_updated=False, status='DIAGNOSTIC_COMPLETE', seconds=time.perf_counter()-start)
                    write(folder/'result.json', result)
                    completed.append(rid)
                    print('SOURCE_ROW '+json.dumps(dict(row_id=rid, count=27000, accuracy=result['mean']['correct'], agreement=result['mean']['agreement'], embedding_distance=result['mean']['features_unit_distance'])), flush=True)
            del model, before, payload
            torch.cuda.empty_cache()
    if actual_flags() != initial_flags or len(completed) != 56:
        raise ValueError('Backend flags not restored or diagnostic matrix incomplete')
    write(output/'completion.json', dict(status='DIAGNOSTIC_MATRIX_COMPLETE', rows=completed, complete_rows=56,
        full_source_V_count_per_row=27000, full_source_cells_per_row=90, target_access=False, model_updated=False,
        backend_flags_restored=True, default_clean='N/A: frozen source diagnostic; original completed predictions remain fixed'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--device', default='cuda:0')
    a = p.parse_args(); main(a.output, a.device)
