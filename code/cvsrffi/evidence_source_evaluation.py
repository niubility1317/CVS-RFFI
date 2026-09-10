"""Frozen source-only V assessment; no target loader or target fitting interface."""
import json
import time
from pathlib import Path
import torch
from .evidence_decision import ConditionAwareCalibrator, selective_metrics
from .evidence_diagnostics import FixedReadoutBaseline, source_condition_probe


def _collect(model, loader, device, role):
    declared = getattr(getattr(loader, 'dataset', None), 'role', role)
    if declared != role:
        raise ValueError(f'expected {role} source loader, got {declared}')
    rows = {}
    evidence_present = None
    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError('source batches require (x, labels, optional metadata)')
        x, labels = batch[0].to(device), batch[1].long()
        meta = next((item for item in batch[2:] if isinstance(item, dict)), {})
        if 'role' in meta:
            roles = [meta['role']] if isinstance(meta['role'], str) else list(meta['role'])
            if any(value != role for value in roles):
                raise ValueError('batch role violates source evaluation contract')
        out = model(x, y_tx=None, grl_lambda=0., return_aux=True)
        evidence = out.get('evidence')
        if evidence_present is not None and evidence_present != (evidence is not None):
            raise ValueError('mixed evidence and H0 batches')
        evidence_present = evidence is not None
        selected = dict(logits=out['tx_logits'], labels=labels,
                        z=evidence['z'] if evidence is not None else out['z_id'])
        if evidence is not None:
            for key in ('mahalanobis','observed_count','quality','state','observed','state_in_domain'):
                selected[key] = evidence[key]
        if 'rx_i' in meta:
            selected['rx'] = torch.as_tensor(meta['rx_i']).long()
        for key, value in selected.items():
            rows.setdefault(key, []).append(value.detach().cpu())
        # Neither covariance nor the full forward result escapes this batch.
        del out, evidence
    if not rows:
        raise ValueError(f'empty {role} loader')
    result = {key:torch.cat(parts) for key,parts in rows.items()}
    if 'rx' in result and len(result['rx']) != len(result['labels']):
        del result['rx']  # Partial metadata must not be reported as a full probe.
    return result


def _metric_summary(metrics):
    return {key:value for key,value in metrics.items() if key != 'risk_coverage'}


def evaluate_source(model, source_loader, val_loader, device, output_dir):
    """Evaluate all supplied L_s/V rows with a frozen model; fit only controls/V calibration.

    Caller must supply contract-validated source loaders. V diagnostics are in-sample
    calibration readouts, not unbiased risk estimates or target selection signals.
    Parameters, buffers, and nested training flags are preserved. CPU storage is
    O(ND+NC); full covariance tensors are only alive within a forward batch.
    """
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {name:folder/name for name in ('source_evaluation.json','source_scores.pt','source_curves.json','source_calibrator.json','source_readouts.pt')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('source evaluation artifacts already exist')
    started = time.perf_counter()
    modes = [(module,module.training) for module in model.modules()]
    versions = {key:value._version for key,value in model.state_dict().items()}
    model.eval()
    try:
        with torch.no_grad():
            train = _collect(model,source_loader,device,'L_s')
            val = _collect(model,val_loader,device,'V')
        extraction_seconds = time.perf_counter()-started
        summary = dict(schema='core90_source_evaluation_v1', role='V', fit_role='L_s',
                       count_train=len(train['labels']),count_val=len(val['labels']),
                       target_access=False, scientific_promotion_claim=False,
                       interpretation='source V calibration and diagnostics on the same V; descriptive, not independent risk certification',
                       missing_diagnostics=[])
        curves = {}
        scores = {key:value for key,value in val.items() if key != 'z'}
        accepted = torch.ones(len(val['labels']),dtype=torch.bool)
        raw = selective_metrics(val['logits'],val['labels'],accepted)
        summary['raw_head'] = _metric_summary(raw)
        curves['raw_head'] = raw['risk_coverage']
        if 'mahalanobis' in val:
            calibrator = ConditionAwareCalibrator().fit(val['logits'],val['mahalanobis'],
                val['observed_count'],val['quality'],val['labels'],role='V',state_coverage=val['state_in_domain'])
            decision = calibrator.predict(val['logits'],val['mahalanobis'],val['observed_count'],
                val['quality'],state_coverage=val['state_in_domain'])
            metrics = selective_metrics(val['logits']/calibrator.temperature,val['labels'],decision['accepted'])
            summary['calibrated_head'] = _metric_summary(metrics)
            summary['calibration'] = dict(temperature=calibrator.temperature,groups=len(calibrator.groups),
                available_fraction=float(decision['calibration_available'].float().mean()),
                state_in_domain_fraction=float(val['state_in_domain'].float().mean()),
                statuses={key:decision['status'].count(key) for key in ('identify','defer','model_mismatch_candidate')})
            curves['calibrated_head'] = metrics['risk_coverage']
            scores['decision'] = {key:value for key,value in decision.items() if torch.is_tensor(value)}
            paths['source_calibrator.json'].write_text(json.dumps(calibrator.state_dict(),indent=2)+'\n',encoding='utf-8')
        else:
            summary['missing_diagnostics'].append('H0 has no observed covariance or quality head: no invented Gaussian consistency calibration')
        summary['baselines'] = {}
        readouts = {}
        for kind in ('cosine','diagonal_gaussian','linear_ridge'):
            baseline = FixedReadoutBaseline(train['z'],train['labels'],role='L_s',kind=kind)
            if not torch.isin(val['labels'],baseline.classes).all():
                raise ValueError('V has classes absent from source L_s')
            logits = baseline.predict(val['z'])
            local_labels = torch.searchsorted(baseline.classes,val['labels'])
            metrics = selective_metrics(logits,local_labels,accepted)
            summary['baselines'][kind] = _metric_summary(metrics)
            curves[kind] = metrics['risk_coverage']
            scores['baseline_'+kind] = logits
            scores['baseline_classes'] = baseline.classes
            if 'mahalanobis' not in val:
                # Export learned closed-form classifier parameters only. No
                # source examples, covariance, or separate means are retained.
                if kind == 'cosine':
                    weight = torch.nn.functional.normalize(baseline.means,dim=-1)
                    bias = weight.new_zeros(len(weight))
                elif kind == 'diagonal_gaussian':
                    weight = baseline.means / baseline.var
                    bias = -.5*(baseline.means.square()/baseline.var).sum(-1)
                else:
                    weight, bias = baseline.weight[:-1].T, baseline.weight[-1]
                readouts[kind] = dict(weight=weight.detach().cpu(),bias=bias.detach().cpu(),
                    normalize_input=torch.tensor(kind=='cosine'),classes=baseline.classes.cpu())
        if readouts:
            torch.save({'version':torch.tensor(1),'controls':readouts},paths['source_readouts.pt'])
        summary['condition_probes'] = {}
        if 'quality' in train and 'quality' in val:
            for feature in ('state','quality','observed'):
                a,b = train[feature].float(),val[feature].float()
                for label in ('labels','rx'):
                    name = feature+'_'+('TX' if label=='labels' else 'RX')
                    if label not in train or label not in val:
                        summary['missing_diagnostics'].append(name+': complete RX metadata unavailable')
                        continue
                    try:
                        summary['condition_probes'][name] = source_condition_probe(a,train[label],b,val[label])
                    except ValueError as error:
                        summary['missing_diagnostics'].append(name+': '+str(error))
        else:
            summary['missing_diagnostics'].append('state/quality/mask probes unavailable for H0')
        if any(value._version != versions[key] for key,value in model.state_dict().items()):
            raise RuntimeError('frozen source evaluation mutated model state')
        summary['resources'] = dict(extraction_seconds=extraction_seconds,total_seconds=time.perf_counter()-started,
                                   rows_per_second=(len(train['labels'])+len(val['labels']))/max(extraction_seconds,1e-9),
                                   covariance_retained=False,device=str(device))
        torch.save(scores,paths['source_scores.pt'])
        paths['source_curves.json'].write_text(json.dumps(curves,indent=2)+'\n',encoding='utf-8')
        summary['artifacts'] = {key:str(path) for key,path in paths.items() if path.exists() or key=='source_evaluation.json'}
        paths['source_evaluation.json'].write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        return summary
    finally:
        for module,training in modes:
            module.training = training
