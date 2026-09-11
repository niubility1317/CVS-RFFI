"""Bounded source-only mechanism evaluation; never a target selection path."""
from contextlib import contextmanager
from dataclasses import replace
import random
import numpy as np
import torch
from .schema import records_from_dataset
from .sampler import CrossBlockSampler
from .roles import make_roles
from .statistics import query_cell_targets
from .tensor_ops import response_loss


@contextmanager
def fixed_source_eval(modules, seed):
    """Restore individual training flags and every framework RNG even on failure."""
    unique = {id(m):m for root in modules for m in root.modules()}
    flags = {key:m.training for key,m in unique.items()}
    py, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        for m in unique.values():
            m.training = False
        for m in unique.values():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and not m.track_running_stats:
                raise ValueError('strict donor evaluation requires fixed BatchNorm running statistics')
        with torch.no_grad():
            yield
    finally:
        for key,m in unique.items():
            m.training = flags[key]
        random.setstate(py)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def raw_received_iq(dataset, index):
    """Read physical received samples before RMS centering/normalization or augmentation."""
    if hasattr(dataset,'raw_iq'):
        raw = torch.as_tensor(dataset.raw_iq(index),dtype=torch.float32)
        if raw.ndim != 2 or raw.shape[0] != 2 or raw.shape[1] < 1:
            raise ValueError('raw_iq hook must return received [2,T] samples')
        return raw
    base = getattr(dataset,'base',dataset)
    if not hasattr(base,'data'):
        raise ValueError('raw received IQ unavailable; normalized model input is not a raw target')
    from dataset_wisig import _pad_or_crop_2t, _safe_to_torch_float_tensor
    row = dataset.index[index]
    raw = np.asarray(base.data[row.tx_i][row.rx_i][row.day_i][row.eq_i][row.sig_i],dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] < 1:
        raise ValueError('WiSig physical record must have shape [T,2]')
    if base.crop_mode not in ('center','left'):
        raise ValueError('raw statistics require fixed center/left crop; random crop unsupported')
    raw = _pad_or_crop_2t(raw.T,int(base.out_len),mode=base.crop_mode)
    return _safe_to_torch_float_tensor(raw)


def event_metadata_from_records(records, windows):
    if not windows or any(r.event_id is None or str(r.event_id)=='' for r in records):
        raise ValueError('event statistics unavailable without real event IDs and fixed protocol windows')
    return dict(event_ids=[r.event_id for r in records],windows=windows)


def evaluate_source_response(model, readouts, predictor, statistics, normalizer,
                             dataset, config, device, domain_label_map, *, source_role):
    """Evaluate deterministic complete blocks on an explicitly source validation subset.

    Config: P/Q/K, source_eval_max_blocks, data_seed, scheduler_candidate_limit.
    Constant baseline is the frozen source-training location (zero standardized).
    Changed-donor tests hold query TX/RX fixed and use separate donor TX subsets.
    """
    if source_role != 'source_validation':
        raise ValueError('mechanism evaluation accepts only explicit source_validation')
    legal = ('ssdg_source_v_cal','ssdg_source_v_select')
    records = records_from_dataset(dataset,allowed_split_sources=legal)
    if not bool(normalizer.fitted):
        raise ValueError('source-training normalization must already be frozen')
    if int(statistics.input_length) < 1:
        raise ValueError('statistics input length must be fixed on source training before validation')
    for ds in (dataset,getattr(dataset,'base',dataset)):
        if getattr(ds,'transform',None) is not None:
            raise ValueError('source mechanism evaluation requires unaugmented clean dataset')
    if statistics.family == 'event':
        if any(r.event_id is None for r in records) or not config.get('event_windows'):
            raise ValueError('real event IDs and fixed protocol event windows required')
    p,q,k = int(config.get('P',4)),int(config.get('Q',4)),int(config.get('K',2))
    limit = int(config.get('source_eval_max_blocks',8))
    seed = int(config.get('data_seed',1337))
    role_policy = 'balanced_partitions_v2' if config.get('implementation_version', 1) == 2 else 'legacy_halves_v1'
    role_count = 36 if role_policy == 'balanced_partitions_v2' else 4
    if limit < 1:
        raise ValueError('source_eval_max_blocks must be positive')
    result = dict(source_role=source_role,split_source=dataset.split_source,valid_blocks=0,
                  valid_role_tasks=0,independent_query_tasks=0,changed_donor_tasks=0,
                  shuffled_rx_tasks=0,records_read=0,backbone_forwards=0,
                  response_mse=None,constant_mse=None,shuffled_rx_mse=None,
                  shuffled_reference_mse=None,changed_donor_reference_mse=None,
                  changed_donor_tx_mse=None,interaction_mse=None,per_rotation={},
                  constant_definition='frozen_source_training_location',status='UNAVAILABLE',
                  role_policy=role_policy,predictor_forwards=0,readout_forwards=0)
    result['skipped_blocks'],result['skipped_roles'],result['skip_reasons'] = 0,0,{}
    def skipped(reason,kind):
        result[kind] += 1
        result['skip_reasons'][reason] = result['skip_reasons'].get(reason,0)+1
    totals = {key:[] for key in ('response','constant','shuffle','changed','interaction','shuffle_base','changed_base')}
    unique_records = set()
    with fixed_source_eval((model,readouts,predictor,statistics,normalizer),seed):
        sampler = CrossBlockSampler(records,P=p,Q=q,k_menu=(k,),batch_size=p*q*k,seed=seed,
                   max_candidates=int(config.get('scheduler_candidate_limit',512)),steps_per_epoch=limit,
                   role_policy=role_policy)
        result['candidate_count'] = len(sampler.candidates)
        result['candidate_search_complete'] = sampler.candidate_search_complete
        if not sampler.candidates:
            result['reason'] = 'no_feasible_complete_source_validation_block'
            return result
        for _ in range(limit):
            plan = sampler.next_batch()
            if not plan.blocks:
                skipped(plan.skip_reason or 'no_feasible_block','skipped_blocks')
                continue
            block = plan.blocks[0]
            block.validate()
            valid_roles = []
            for rotation in range(role_count):
                role = make_roles(block.candidate.tx_ids,block.candidate.rx_ids,rotation,policy=role_policy)
                try:
                    replace(block,roles=role).validate()
                except ValueError as error:
                    skipped(str(error),'skipped_roles')
                    continue
                valid_roles.append(role)
            if not valid_roles:
                skipped('no_isolated_role_template','skipped_blocks')
                continue
            loaded = [dataset[r.index] for r in block.records]
            iq = torch.stack([torch.as_tensor(row[0]) for row in loaded]).to(device)
            raw = torch.stack([raw_received_iq(dataset,r.index) for r in block.records]).to(device)
            domains = torch.tensor([domain_label_map.get(int(row[2]),-1) for row in loaded],device=device)
            out = model(iq,y_tx=None,grl_lambda=0.,return_aux=True,domain_labels=domains)
            zi,zd = out['z_id'].reshape(p,q,k,-1),out['z_dom'].reshape(p,q,k,-1)
            if not torch.isfinite(zi).all() or not torch.isfinite(zd).all():
                raise ValueError('nonfinite source validation features')
            event = None if statistics.family != 'event' else event_metadata_from_records(block.records,config['event_windows'])
            target = query_cell_targets(statistics,raw.reshape(p,q,k,*raw.shape[1:]),normalizer,event)
            c = block.candidate
            for role in valid_roles:
                rotation = role.rotation
                qt,dt = [c.tx_ids.index(v) for v in role.query_tx],[c.tx_ids.index(v) for v in role.donor_tx]
                qr,dr = [c.rx_ids.index(v) for v in role.query_rx],[c.rx_ids.index(v) for v in role.donor_rx]
                t,d = readouts(zi,zd,qt,qr,dt,dr)
                prediction = predictor(t,d)
                result['predictor_forwards'] += 1
                result['readout_forwards'] += 1
                truth = target[qt][:,qr]
                loss,diag = response_loss(prediction,truth)
                value = float(loss)
                totals['response'].append(value)
                totals['constant'].append(float(truth.square().mean()))
                result['per_rotation'].setdefault(str(rotation),[]).append(value)
                result['valid_role_tasks'] += 1
                if role.independent_query_rectangle:
                    result['independent_query_tasks'] += 1
                    totals['interaction'].append(float(diag['interaction_error']))
                if len(qr)>1:
                    shuffled = predictor(t,d.roll(1,0))
                    result['predictor_forwards'] += 1
                    totals['shuffle'].append(float((shuffled-truth).square().mean()))
                    totals['shuffle_base'].append(value)
                    result['shuffled_rx_tasks'] += 1
                if len(dt)>1:
                    for alternate in (dt[::2],dt[1::2]):
                        at,ad = readouts(zi,zd,qt,qr,alternate,dr)
                        totals['changed'].append(float((predictor(at,ad)-truth).square().mean()))
                        result['predictor_forwards'] += 1
                        result['readout_forwards'] += 1
                        totals['changed_base'].append(value)
                        result['changed_donor_tasks'] += 1
            result['valid_blocks'] += 1
            result['records_read'] += c.size
            unique_records.update(r.physical_sample_id for r in block.records)
            result['backbone_forwards'] += 1
    def mean(values):
        return sum(values)/len(values) if values else None
    for field,key in (('response_mse','response'),('constant_mse','constant'),
                      ('shuffled_rx_mse','shuffle'),('changed_donor_tx_mse','changed'),('interaction_mse','interaction'),
                      ('shuffled_reference_mse','shuffle_base'),('changed_donor_reference_mse','changed_base')):
        result[field] = mean(totals[key])
    result['per_rotation'] = {key:mean(values) for key,values in result['per_rotation'].items()}
    if not result['valid_role_tasks']:
        result['reason'] = 'no_isolated_complete_source_validation_tasks'
        return result
    result['status'],result['reason'] = 'VERIFIED','source_mechanism_metrics_only'
    result['constant_improvement'] = result['constant_mse']-result['response_mse']
    result['unique_physical_records_read'] = len(unique_records)
    result['reduced_donor_count_mse'] = result['changed_donor_tx_mse']
    result['reduced_donor_count_tasks'] = result['changed_donor_tasks']
    result['changed_donor_legacy_field_definition'] = 'reduced_donor_count_not_equal_size_reuse'
    return result


def evaluate_source_response_v2(model, readouts, predictor, statistics, normalizer,
                                dataset, config, device, domain_label_map, *,
                                source_role, source_train_dataset):
    """Matched source fits and whole-pool equal-size donor reuse diagnostics.

    This is an explicit separate V2 API. The original pretrained predictor and
    independently fitted heads are separately named; neither implies that a
    source necessity/update-benefit gate passed. No target or hidden U_s labels.
    """
    from .source_baselines import (build_donor_reuse_tasks, fit_source_baselines,
                                   prediction_errors)
    if source_role != 'source_validation':
        raise ValueError('V2 accepts only source_validation')
    train_records = records_from_dataset(source_train_dataset,
                                         allowed_split_sources=('ssdg_labeled_tx_visible',))
    eval_records = records_from_dataset(dataset,
        allowed_split_sources=('ssdg_source_v_cal', 'ssdg_source_v_select'))
    if {r.physical_sample_id for r in train_records} & {r.physical_sample_id for r in eval_records}:
        raise ValueError('source fit and evaluation physical IDs overlap')
    if not bool(normalizer.fitted) or int(statistics.input_length) < 1:
        raise ValueError('source statistics and normalization must already be frozen')
    for current in (source_train_dataset, dataset):
        for ds in (current, getattr(current, 'base', current)):
            if getattr(ds, 'transform', None) is not None:
                raise ValueError('V2 descriptor audit requires unaugmented source data')
    k = int(config.get('K', 2))
    limit = int(config.get('source_eval_max_blocks', 8))
    # The fit budget is explicit: an absent budget does not silently become a
    # supposedly preregistered or optimized scientific setting.
    if 'source_baseline_fit_steps' not in config or 'source_baseline_fit_lr' not in config:
        raise ValueError('freeze source_baseline_fit_steps and source_baseline_fit_lr explicitly')
    train_plans = build_donor_reuse_tasks(train_records, k=k, max_tasks=limit)
    eval_plans = build_donor_reuse_tasks(eval_records, k=k, max_tasks=limit)
    result = {'version': 'source_response_v2', 'source_role': source_role,
              'status': 'N/A', 'necessity_gate_passed': None,
              'reuse': {}, 'cost': {'backbone_forwards': 0, 'model_input_records': 0,
                                    'raw_statistic_records': 0, 'predictor_forwards': 0},
              'fit_split': source_train_dataset.split_source, 'eval_split': dataset.split_source}
    result['frozen_parameter_counts'] = {'backbone': sum(p.numel() for p in model.parameters()),
                                          'readouts': sum(p.numel() for p in readouts.parameters()),
                                          'pretrained_predictor': sum(p.numel() for p in predictor.parameters())}
    unique_fit, unique_eval = set(), set()
    seed = int(config.get('data_seed', 1337))

    def descriptors(ds, task, branch, unique):
        task.validate()
        rows = task.donor_records[branch]
        loaded = [ds[r.index] for r in rows]
        iq = torch.stack([torch.as_tensor(v[0]) for v in loaded]).to(device)
        domains = torch.tensor([domain_label_map.get(int(v[2]), -1) for v in loaded], device=device)
        output = model(iq, y_tx=None, grl_lambda=0., return_aux=True, domain_labels=domains)
        tx = torch.stack([readouts.tx(output['z_id'][[i for i, r in enumerate(rows)
                         if r.tx_id == t and r.rx_id in task.donor_rx[branch]]]).mean(0)
                          for t in task.query_tx])
        rx = torch.stack([readouts.rx(output['z_dom'][[i for i, r in enumerate(rows)
                         if r.rx_id == r_id and r.tx_id in task.donor_tx[branch]]]).mean(0)
                          for r_id in task.query_rx])
        if not bool(torch.isfinite(tx).all() and torch.isfinite(rx).all()):
            raise ValueError('nonfinite source descriptors')
        result['cost']['backbone_forwards'] += 1
        result['cost']['model_input_records'] += len(rows)
        unique.update(r.physical_sample_id for r in rows)
        return {'tx': tx.detach(), 'rx': rx.detach()}

    def targets(ds, task, unique):
        rows = task.query_records
        raw = torch.stack([raw_received_iq(ds, r.index) for r in rows]).to(device)
        event = None if statistics.family != 'event' else event_metadata_from_records(rows, config.get('event_windows'))
        grid = raw.reshape(len(task.query_tx), len(task.query_rx), task.k, *raw.shape[1:])
        truth = query_cell_targets(statistics, grid, normalizer, event)
        result['cost']['raw_statistic_records'] += len(rows)
        unique.update(r.physical_sample_id for r in rows)
        return truth

    fit_tasks, validation_tasks = [], []
    with fixed_source_eval((model, readouts, predictor, statistics, normalizer), seed):
        for kind, entry in train_plans.items():
            for task in entry['tasks']:
                target = targets(source_train_dataset, task, unique_fit)
                for branch in (0, 1):
                    fit_tasks.append(dict(descriptors(source_train_dataset, task, branch, unique_fit), target=target))
        for kind, entry in eval_plans.items():
            report = {'status': entry['status'], 'reason': entry['reason'],
                      'query_shape': entry['query_shape'],
                      'independent_query_interaction': entry['independent_query_interaction'],
                      'tasks': []}
            result['reuse'][kind] = report
            for task in entry['tasks']:
                truth = targets(dataset, task, unique_eval)
                branches, predictions = [], []
                for branch in (0, 1):
                    desc = dict(descriptors(dataset, task, branch, unique_eval), target=truth)
                    validation_tasks.append(desc)
                    prediction = predictor(desc['tx'], desc['rx'])
                    predictions.append(prediction)
                    # Fixed cyclic permutations: query and conditions unchanged.
                    sh_tx = predictor(desc['tx'].roll(1, 0), desc['rx']) if len(desc['tx']) > 1 else None
                    sh_rx = predictor(desc['tx'], desc['rx'].roll(1, 0)) if len(desc['rx']) > 1 else None
                    result['cost']['predictor_forwards'] += 1 + int(sh_tx is not None) + int(sh_rx is not None)
                    branches.append({'pretrained_full': prediction_errors(prediction, truth),
                                     'constant_frozen_source_location': prediction_errors(torch.zeros_like(truth), truth),
                                     'shuffled_tx': prediction_errors(sh_tx, truth) if sh_tx is not None else None,
                                     'shuffled_rx': prediction_errors(sh_rx, truth) if sh_rx is not None else None,
                                     'donor_tx': list(task.donor_tx[branch]),
                                     'donor_rx': list(task.donor_rx[branch]),
                                     'donor_records': len(task.donor_records[branch]),
                                     'donor_physical_ids': [r.physical_sample_id for r in task.donor_records[branch]]})
                report['tasks'].append({'query_tx': list(task.query_tx), 'query_rx': list(task.query_rx),
                                        'query_physical_ids': [r.physical_sample_id for r in task.query_records],
                                        'day': task.day_id, 'condition': task.condition_id, 'K': task.k,
                                        'branches': branches,
                                        'prediction_reuse_mse': float((predictions[0]-predictions[1]).square().mean())})
    if fit_tasks and validation_tasks:
        # fixed_source_eval has exited: gradient-enabled auxiliary fits only.
        result['matched_fits'] = fit_source_baselines(predictor, fit_tasks, validation_tasks,
            steps=int(config['source_baseline_fit_steps']), lr=float(config['source_baseline_fit_lr']), seed=seed,
            fit_physical_ids=unique_fit, eval_physical_ids=unique_eval,
            fit_role='source_train', eval_role='source_validation')
        result['cost']['auxiliary_fit_forwards'] = 3 * int(config['source_baseline_fit_steps'])
        result['cost']['auxiliary_eval_forwards'] = 3 * len(validation_tasks)
        result['status'] = 'VERIFIED'
        result['scope'] = 'source_metrics_only_not_necessity_or_update_value_acceptance'
    else:
        result['reason'] = 'insufficient_independent_source_fit_or_validation_tasks'
    result['cost']['fit_unique_physical_records'] = len(unique_fit)
    result['cost']['eval_unique_physical_records'] = len(unique_eval)
    result['identity_risk'] = {'status': 'N/A', 'reason': 'response_MSE_is_not_identity_CE_use_disposable_counterfactual_with_independent_source_LEO_probe'}
    return result
