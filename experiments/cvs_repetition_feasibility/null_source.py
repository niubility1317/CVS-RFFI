"""Fixed null-reference14 feasibility on original source L; no learned model."""
import json
import os
import sys
import time
from pathlib import Path

import torch

from .budget_loss import budget_loss
from .metrics import summarize, qualification
from .null_observation import null_evidence
from .relations import measure

RUN = '20261014-phase1-null-budget-sourceL-manysig-m1-r01'
RELEASE = 'cvs_null_budget_sourceL_20261014_r01'
PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'


def source_qualification(metrics, geometry, budgets):
    base = qualification(metrics, geometry)
    checks = dict(base['checks'])
    checks['budget_signal_weight'] = budgets['overall']['visible_mean']['weight_mean'] >= .05
    checks['budget_active_coverage'] = budgets['overall']['visible_mean']['active_fraction'] >= .05
    checks['all_RX_budget_active'] = all(
        budgets['RX/' + str(rx)]['visible_mean']['active_fraction'] >= .01
        for rx in (1, 3, 4, 6, 8))
    return dict(passed=all(checks.values()), checks=checks,
                source_only=True, target_access=False, parameter_scan=False,
                rule=base['rule'] + ' Additionally: visible-mean budget weight mean>=.05, '
                     'active fraction>=5% overall and>=1% in each RX. These are fixed '
                     'necessary source signal-availability conditions, not DG evidence.')


def budget_census(target, predictions, signal, budget, receivers, valid):
    groups = {'overall': torch.ones_like(valid)}
    groups.update({'RX/' + str(rx): receivers == rx for rx in (1, 3, 4, 6, 8)})
    records = {}
    tensors = {}
    for arm, prediction in predictions.items():
        result = budget_loss(prediction, target, signal, budget, role='L_s')
        tensors[arm] = {k: v.detach() for k, v in result['tensors'].items()}
        for name, mask in groups.items():
            selected = mask & valid
            full = int(mask.sum())
            count = int(selected.sum())
            values = tensors[arm]
            active = selected & (values['a'] > 0) & (values['w'] > 0)
            rec = dict(count=full, valid=count, active_count=int(active.sum()),
                       active_fraction=int(active.sum()) / full,
                       fulfilled_count=int((selected & (values['MSE'] <= budget)).sum()),
                       weight_mean=float(values['w'][selected].mean()) if count else None,
                       signal_mean=float(signal[selected].mean()) if count else None,
                       budget_mean=float(budget[selected].mean()) if count else None,
                       weighted_loss=float(values['per_packet_loss'][selected].sum()) / full,
                       maximum_packet_loss=float(values['per_packet_loss'][selected].max()) if count else None,
                       normalization='complete group count; zero-visible packets counted with zero contribution')
            records.setdefault(name, {})[arm] = rec
    return records, tensors


def run(request):
    expected = PROJECT + '/runs/' + RUN + '/L6300/source'
    if (request['run_id'] != RUN or request['output_root'] != expected
            or request['source_role'] != 'L_s'
            or request.get('observation_kind') != 'null_reference14'):
        raise ValueError('Only fixed null-reference14 source-L6300 and owned output root')
    out = Path(expected)
    if out.exists():
        raise FileExistsError('Preserve existing source diagnostic; no relaunch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    runtime = dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, device='cpu',
                   GPU_allocations=0, analysis_code_commit=request['analysis_code_commit'], run_id=RUN)
    (out / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
    print('RUNTIME ' + json.dumps(runtime), flush=True)
    smoke = null_evidence(torch.zeros(2, 2, 256))
    if smoke['valid'].any() or smoke['target'].shape != (2, 14) or smoke['masked_input'].any():
        raise ValueError('No-query zero-IQ observation smoke failed')
    from experiments.cvs_pa_recomposition import design as d
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    c = d.config(next(r for r in d.rows() if r['row_id'] == 'ZERO-s2026092701'))
    packets = {k: [] for k in ('target', 'valid', 'signal', 'budget', 'variance', 'cycles')}
    predictions = {k: [] for k in ('copy_cycle1', 'visible_mean', 'known_reference')}
    labels, receivers, days, physical_ids = [], [], [], []
    with installed(c) as native:
        context = native._build_ssdg_wisig_data(configure(d.make_args(c, 'cpu')), torch.device('cpu'))
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        if context['named_test_loaders'] or role_ids_from_native(context) != d.read(d.SOURCE)['role_ids']:
            raise ValueError('Original exact source-only data roles required')
        dataset = context['train_loader'].dataset
        if len(dataset) != 6300 or getattr(dataset, 'hide', True):
            raise ValueError('Original visible source L6300 only')
        with torch.no_grad():
            for begin in range(0, 6300, 128):
                samples = [dataset[i] for i in range(begin, min(begin + 128, 6300))]
                result = null_evidence(torch.stack([s[0] for s in samples]))
                if result['masked_input'][:, :, 120:140].any():
                    raise ValueError('Hidden period entered masked input')
                for k in packets:
                    packets[k].append(result[k])
                for arm in predictions:
                    key = 'zero_reference' if arm == 'known_reference' else arm
                    predictions[arm].append(result['predictions'][key])
                # Prediction/calibration never consume labels or grouping metadata.
                for _, y, _, meta in samples:
                    labels.append(int(y))
                    receivers.append(int(meta['rx_i']))
                    days.append(int(meta['day_i']))
                    physical_ids.append([int(meta[k]) for k in ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')])
                if begin % 1024 == 0:
                    print(f'SOURCE_L_SEEN {min(begin + len(samples), 6300)}/6300', flush=True)
    packets = {k: torch.cat(v) for k, v in packets.items()}
    predictions = {k: torch.cat(v) for k, v in predictions.items()}
    labels, receivers, days = map(torch.tensor, (labels, receivers, days))
    metrics = summarize(packets['valid'], packets['target'], predictions,
                        labels, receivers, days, observation_kind='null_reference14')
    opaque_ids = [json.dumps(v, separators=(',', ':')) for v in physical_ids]
    geometry = measure(predictions['visible_mean'], labels.tolist(), receivers.tolist(), days.tolist(),
                       opaque_ids, packets['valid'], role='L_s', vector_kind='visible_mean_null_reference14')
    budgets, budget_tensors = budget_census(packets['target'], predictions, packets['signal'],
                                           packets['budget'], receivers, packets['valid'])
    qualified = source_qualification(metrics, geometry['summary'], budgets)
    summary = dict(status='SOURCE_NULL_BUDGET_FEASIBILITY_COMPLETE', run_id=RUN,
                   analysis_code_commit=request['analysis_code_commit'], source_role='L_s',
                   source_count=6300, source_IQ_packet_reads=6300, network_IQ_forwards=0,
                   noquery_fixture_packets=2, checkpoint_sources=[], model_or_decoder_fitting=False,
                   head_fit_updates=0, optimizer_updates=0, EMA_updates=0, V_gradient_updates=0,
                   U_IQ_reads=0, U_truth_read=False, target_access=False,
                   metrics=metrics['records'], metric_normalization=metrics['normalization'],
                   relation_summary=geometry['summary'], budget_records=budgets, qualification=qualified,
                   primitive=result['summary'], runtime=runtime, wall_seconds=time.perf_counter() - started,
                   limitation='Seen-source null-reference14 observation, not deployed joint160, TX-exclusive '
                              'hardware, proven noise floor, decoder/backbone efficacy or new-RX DG.')
    summary['primitive']['input_count'] = 6300
    summary['primitive']['valid_count'] = int(packets['valid'].sum())
    summary['primitive']['zero_visible_count'] = int((~packets['valid']).sum())
    torch.save(dict(**packets, predictions=predictions, labels=labels, receivers=receivers, days=days,
                    physical_ids=physical_ids, packet_target_energy=metrics['packet_target_energy'],
                    packet_MSE=metrics['packet_MSE'], relations=geometry['tensors'], budget_tensors=budget_tensors),
               out / 'observation_census.pt')
    d.write(out / 'summary.json', summary)
    print('SOURCE_NULL_BUDGET_FEASIBILITY_COMPLETE ' + json.dumps(dict(
        qualification=qualified, overall=summary['metrics']['overall'],
        budget=budgets['overall'], geometry=summary['relation_summary']['overall'])), flush=True)
    return summary
