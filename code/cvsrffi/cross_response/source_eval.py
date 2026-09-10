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
    if limit < 1:
        raise ValueError('source_eval_max_blocks must be positive')
    result = dict(source_role=source_role,split_source=dataset.split_source,valid_blocks=0,
                  valid_role_tasks=0,independent_query_tasks=0,changed_donor_tasks=0,
                  shuffled_rx_tasks=0,records_read=0,backbone_forwards=0,
                  response_mse=None,constant_mse=None,shuffled_rx_mse=None,
                  shuffled_reference_mse=None,changed_donor_reference_mse=None,
                  changed_donor_tx_mse=None,interaction_mse=None,per_rotation={},
                  constant_definition='frozen_source_training_location',status='UNAVAILABLE')
    result['skipped_blocks'],result['skipped_roles'],result['skip_reasons'] = 0,0,{}
    def skipped(reason,kind):
        result[kind] += 1
        result['skip_reasons'][reason] = result['skip_reasons'].get(reason,0)+1
    totals = {key:[] for key in ('response','constant','shuffle','changed','interaction','shuffle_base','changed_base')}
    unique_records = set()
    with fixed_source_eval((model,readouts,predictor,statistics,normalizer),seed):
        sampler = CrossBlockSampler(records,P=p,Q=q,k_menu=(k,),batch_size=p*q*k,seed=seed,
                   max_candidates=int(config.get('scheduler_candidate_limit',512)),steps_per_epoch=limit)
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
            for rotation in range(4):
                role = make_roles(block.candidate.tx_ids,block.candidate.rx_ids,rotation)
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
                    totals['shuffle'].append(float((shuffled-truth).square().mean()))
                    totals['shuffle_base'].append(value)
                    result['shuffled_rx_tasks'] += 1
                if len(dt)>1:
                    for alternate in (dt[::2],dt[1::2]):
                        at,ad = readouts(zi,zd,qt,qr,alternate,dr)
                        totals['changed'].append(float((predictor(at,ad)-truth).square().mean()))
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
    return result
