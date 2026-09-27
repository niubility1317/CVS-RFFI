"""Versioned, independent audit evidence. Missing measurements are never zero."""
from copy import deepcopy

SCHEMA = 'game_audit_v2'
STATUSES = frozenset({'OPTIMIZATION_FAILURE', 'BUDGET_INCONCLUSIVE', 'TRANSFER_FAILURE',
                      'RELIABLE_HIGH_GAP', 'RELIABLE_LOW_GAP', 'UNAVAILABLE'})


def unavailable(reason='not_measured'):
    return dict(status='UNAVAILABLE', quality_pass=False, control_ready=False,
                gap_raw=None, gap_normalized=None, reason_codes=[reason])


def make_evidence_v2(*, observation_id, step, encoder_version, data_valid, coverage_valid,
                     lag=None, cross_tx_readout=None, gradient=None, capability=None):
    return dict(schema=SCHEMA, observation_id=str(observation_id), step=int(step),
                encoder_version=int(encoder_version), data_valid=bool(data_valid),
                coverage_valid=bool(coverage_valid), lag=deepcopy(lag or unavailable()),
                cross_tx_readout=deepcopy(cross_tx_readout or unavailable()),
                gradient=deepcopy(gradient or unavailable()),
                capability=deepcopy(capability or unavailable()))


def require_evidence_v2(evidence):
    if not isinstance(evidence, dict) or evidence.get('schema') != SCHEMA:
        raise ValueError('v2 audit evidence required; legacy v1 cannot control v2')
    for key in ('observation_id', 'step', 'encoder_version', 'data_valid', 'coverage_valid',
                'lag', 'cross_tx_readout', 'gradient', 'capability'):
        if key not in evidence:
            raise ValueError('incomplete v2 audit evidence: ' + key)
    if not evidence['observation_id']:
        raise ValueError('v2 observation_id must be nonempty')
    return evidence
