"""Independently certify MarginJoint support archives, traces and resources.

No candidate fit, forward, score or adjoint is called by this analyzer. Saved
Gaussian geometry, frozen actual B and complete KKT equations are its evidence.
"""
import argparse
from collections import OrderedDict
import csv
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'tools')]
from evaluate_d92_margin_joint_probe import (
    CHANNEL, SCENARIOS, SCOPE, STATUS, SCHEMA, METHOD, PATHS, METRICS, BRANCHES,
    PROBE_CONFIG, COUNTERS, PREPARATION_COUNTERS, STAGE_COUNTERS, PEAK_COUNTERS,
    read, check, csv_record, split_identity, selected_tasks, compact_event,
    assess_paths, pooled_assess, parent_mean, compact_record, json_native,
)
from run_d92_margin_joint_probe import validate_spec, verify_marker, budget_for_spec, command
import summarize_d92_affine_joint_probe as affine_analysis
from summarize_d92_branch_support_probe import jsonlines, check_bind, finite_tree, close
from summarize_d92_registration_diagnostic import _statistics, write_json
from cvsrffi.d92_branch_local_ridge import _distances
from d92_affine_analysis_math import center_kernel_vjp

SUMMARY_STATUS = 'COMPLETE_MARGIN_JOINT_PROBE_VERIFIED'
SUMMARY_SCHEMA = 'd92_margin_joint_support_summary_v1'
QP_WORK_KEYS = (
    'transitions','full_constraint_scans','spectral_checks','compact_snapshot_rebuilds',
    'compact_snapshot_dense_work_units','spectral_cubic_dimension_units','independence_checks',
    'factorization_attempts','factorizations_completed','condition_estimation_calls',
    'triangular_calls','triangular_rhs_columns','triangular_rhs_elements','triangular_dense_work_units')
SUFFIXES = ('factorization_count','triangular_solve_count','triangular_rhs_count',
            'triangular_rhs_element_count','triangular_dense_work_unit_count')
KERNEL_COUNTERS = tuple('margin_qp_'+op+'_'+key for op in ('forward','adjoint') for key in QP_WORK_KEYS)+('completed_factorization_count',)
_array_close = affine_analysis._array_close
_scalar = affine_analysis._scalar
adapted_blocks = affine_analysis.adapted_blocks
dct_initial = affine_analysis.dct_initial


class StateResolver(affine_analysis.StateResolver):
    """Reuse numeric inventory/cache checks; require the independent schema."""
    def __init__(self, root, cache_budget_bytes=64 * 1024 * 1024, max_entries=64):
        self.root = Path(root).resolve()
        self.manifest = read(self.root / 'state_manifest.json')
        check(self.manifest['schema'] == 'd92_margin_joint_state_archive_v1'
              and self.manifest['method'] == METHOD and self.manifest['status'] == 'COMPLETE'
              and self.manifest['prediction_formula'] == 'actual_B_prior_plus_full_support_margin_constrained_affine_residual',
              'Incomplete Margin state archive')
        files = self.manifest['files']; self.refs = {ref['path']: ref for ref in files}
        check(len(files) == len(self.refs) == self.manifest['file_count'], 'Duplicate/missing numeric archive')
        check(self.manifest['total_file_bytes'] == sum(ref['file_bytes'] for ref in files), 'Archive byte inventory mismatch')
        check(type(cache_budget_bytes) is int and cache_budget_bytes >= 0
              and type(max_entries) is int and max_entries >= 0, 'Invalid archive cache budget')
        self.used, self.verified, self.cache = set(), set(), OrderedDict()
        self.cache_budget_bytes, self.max_entries = cache_budget_bytes, max_entries
        self.cache_numeric_bytes = 0; self.cache_sizes = {}; self.file_mtimes = {}
        self.cache_counts = dict(load_count=0, hit_count=0, eviction_count=0, loaded_numeric_bytes=0,
            evicted_numeric_bytes=0, oversized_load_count=0, uncached_load_count=0, clear_count=0,
            cleared_entry_count=0, cleared_numeric_bytes=0, peak_numeric_bytes=0, peak_entries=0)
        self.analysis_work = dict(independent_general_solve_count=0, independent_general_rhs_column_count=0,
            independent_general_rhs_element_count=0, independent_dense_cubic_work_unit_count=0,
            independent_dense_rhs_work_unit_count=0, independent_spectral_diagnostic_count=0,
            independent_spectral_cubic_work_unit_count=0)

    def __call__(self, ref):
        # The core recorder returns the callback identity unchanged, but replaces
        # each array's metadata with its shape/dtype/nbytes view. The archive
        # manifest additionally owns the two measured finiteness annotations.
        # Bind only that exact view to its complete manifest entry; no identity,
        # array inventory or numeric metadata field may be omitted or changed.
        check(isinstance(ref, dict) and ref.get('path') in self.refs,
              'State reference missing from exact archive manifest')
        canonical = self.refs[ref['path']]
        check(set(ref) == set(canonical) and all(ref[key] == canonical[key]
              for key in canonical if key != 'arrays'),
              'State reference identity differs from exact archive manifest')
        check(isinstance(ref['arrays'], dict) and set(ref['arrays']) == set(canonical['arrays']),
              'State reference array inventory differs from exact archive manifest')
        check(canonical['failed_numeric_state'] is False,
              'Failed numeric state cannot appear in a complete archive')
        for name, saved in canonical['arrays'].items():
            check(set(saved) == {'shape', 'dtype', 'nbytes', 'all_finite', 'nonfinite_count'}
                  and saved['all_finite'] is True and type(saved['nonfinite_count']) is int
                  and saved['nonfinite_count'] == 0,
                  'Complete archive finiteness metadata mismatch')
            basic = {key: saved[key] for key in ('shape', 'dtype', 'nbytes')}
            check(ref['arrays'][name] == saved or ref['arrays'][name] == basic,
                  'State reference numeric metadata differs from exact archive manifest')
        return super().__call__(canonical)


def verify_artifact_inventory(root, marker):
    root = Path(root).resolve(); value = read(root / 'artifact_manifest.json')
    check(value['schema'] == 'd92_margin_joint_artifacts_v1' and value['method'] == METHOD
          and value['status'] == STATUS and marker['artifact_manifest'] == 'artifact_manifest.json', 'Incomplete artifact inventory')
    listed = {item['path']: item['file_bytes'] for item in value['files']}
    actual = {p.relative_to(root).as_posix(): p.stat().st_size for p in root.rglob('*')
              if p.is_file() and p.name != 'artifact_manifest.json'}
    check(len(listed) == len(value['files']) and listed == actual, 'Artifact inventory mismatch')


def _raw(b, a):
    full = np.column_stack((b, a))
    return dict(zip(BRANCHES, (full[:, :160], full[:, 160:256], full[:, 256:416],
                              full[:, 416:576], full[:, 576:736])))


def _radial(distance, tau, gamma, minus_one=False):
    if gamma is None: return np.zeros_like(distance)
    if tau == 0: return np.asarray(distance == 0, dtype=np.float64) - float(minus_one)
    check(tau is not None and tau > 0 and gamma > 0, 'Undefined raw Gaussian scale')
    return np.expm1(-distance / tau) if minus_one else np.exp(-distance / tau)


def head_score(data, raw, *, prefix=''):
    """Independent fixed single-record inference for B, C and the R0 baseline."""
    part = {key[len(prefix):]: value for key, value in data.items() if key.startswith(prefix)} if prefix else data
    # Saved original geometry is the normalized five-branch representation,
    # not a concatenation of the received raw vectors.  The zero adapter is an
    # independent way to rebuild that same fixed representation for every head.
    ob, oa = adapted_blocks(raw, np.zeros((736, 8)))
    if 'U' not in part:
        tau, gamma = _scalar(part, 'tau'), _scalar(part, 'gamma'); n = len(part['alpha']); q = np.full(n, 1 / n)
        train = _radial(_distances(part['original_train_b'], part['original_train_a']), tau, gamma, True)
        scores = np.empty((len(ob), part['alpha'].shape[1]))
        for i in range(len(ob)):
            cross = _radial(_distances(ob[i:i + 1], oa[i:i + 1], part['original_train_b'], part['original_train_a']), tau, gamma, True)
            _, L = affine_analysis._center(train, cross, q, gamma); scores[i] = (L @ part['alpha'])[0]
        _array_close(affine_analysis.head_score(part, raw), scores, 'R0 deployment gauge differs from independent centered function')
        return scores
    b, a = adapted_blocks(raw, part['U']); tau, gamma = _scalar(part, 'tau'), _scalar(part, 'gamma')
    c = part['alpha'].shape[1]; scores = np.empty((len(ob), c))
    for i in range(len(ob)):
        d0 = _distances(ob[i:i + 1], oa[i:i + 1], part['original_train_b'], part['original_train_a'])
        d = d0 if tau == 0 or not np.any(part['U']) else .5 * (d0 + _distances(
            b[i:i + 1], a[i:i + 1], part['adapted_train_b'], part['adapted_train_a']))
        if 'intercept' in part:
            if gamma is None: row = part['intercept'].copy()
            else:
                train = _radial(part['distance'], tau, gamma, True)
                cross = _radial(d, tau, gamma, True)
                _, L = affine_analysis._center(train, cross, part['q'], gamma)
                row = (L @ part['alpha'])[0] + part['intercept']
        else:
            row = part['b'].copy() if gamma is None else gamma * _radial(d,tau,gamma)[0] @ part['alpha'] + part['b']
        scores[i] = row
    if not prefix and 'prior_old_class_indices' in part:
        prior = head_score(part, raw, prefix='prior_B_')
        indices = part['prior_old_class_indices']
        check(indices.dtype.kind in 'iu' and indices.shape == (prior.shape[1],)
              and len(set(indices.tolist())) == len(indices), 'Invalid actual B class map')
        scores[:, indices] += prior
    check(np.isfinite(scores).all(), 'Nonfinite independent scores')
    return scores


def _factor(factor, matrix, message):
    check(factor.shape == matrix.shape and np.all(np.diag(factor) > 0)
          and np.array_equal(factor, np.tril(factor)), message + ' shape/triangular/positive diagonal')
    _array_close(factor @ factor.T, matrix, message, scale=np.linalg.norm(factor) ** 2)


def _solve(matrix,rhs,work=None):
    """Analysis-only general solve; no candidate factor/fit/predict calls."""
    if work is not None:
        width = 1 if rhs.ndim==1 else rhs.shape[1]
        for key,amount in (('independent_general_solve_count',1),
            ('independent_general_rhs_column_count',width),
            ('independent_general_rhs_element_count',len(matrix)*width),
            ('independent_dense_cubic_work_unit_count',len(matrix)**3),
            ('independent_dense_rhs_work_unit_count',len(matrix)**2*width)):
            work[key] = work.get(key,0)+amount
    return np.linalg.solve(matrix,rhs)


def _pairs(y,old,c):
    rows = np.repeat(np.arange(len(old)),c-1)
    truth = np.repeat(y[old],c-1)
    other = np.concatenate([np.delete(np.arange(c),y[i]) for i in old])
    return rows,truth,other


def _difference(value,pairs):
    rows,truth,other = pairs
    return value[rows,truth]-value[rows,other]


def _scatter(mu,pairs,m,c):
    result = np.zeros((m,c)); rows,truth,other = pairs
    np.add.at(result,(rows,truth),mu); np.add.at(result,(rows,other),-mu)
    return result


def _working(P,pairs,W):
    rows,y,j = (value[W] for value in pairs)
    inner = (y[:,None]==y[None,:]).astype(float)-(y[:,None]==j[None,:]).astype(float)
    inner -= (j[:,None]==y[None,:]); inner += (j[:,None]==j[None,:])
    return P[np.ix_(rows,rows)]*inner


def verify_margin_head(data,audit=None,analysis_work=None):
    """All-row primal/dual certificate, including singular K; never optimize."""
    K,M,y,old = (np.asarray(data[key]) for key in ('K','M','labels','old_indices'))
    n,c = M.shape; m = len(old)
    check(n>0 and c>=2 and m>0 and K.shape==(n,n) and np.array_equal(K,K.T)
        and y.shape==(n,) and y.dtype.kind in 'iu' and old.dtype.kind in 'iu'
        and np.array_equal(old,np.unique(old)) and np.all((old>=0)&(old<n))
        and np.all((y>=0)&(y<c)),'Invalid full physical margin head')
    check(all(np.isfinite(v).all() for v in (K,M,data['alpha'],data['b'])),'Nonfinite margin certificate')
    spectrum = np.linalg.eigvalsh(K)
    if analysis_work is not None:
        analysis_work['independent_spectral_diagnostic_count'] = analysis_work.get('independent_spectral_diagnostic_count',0)+1
        analysis_work['independent_spectral_cubic_work_unit_count'] = analysis_work.get('independent_spectral_cubic_work_unit_count',0)+n**3
    eps = np.finfo(float).eps; k = 16*(n+1); spectral_tol = k*eps/(1-k*eps)*max(1.,np.linalg.norm(K,1))
    check(spectrum[0]>=-spectral_tol,'Independent raw kernel is not PSD')
    A = K+np.eye(n); e = np.ones(n)
    H = np.block([[A,e[:,None]],[e[None,:],np.zeros((1,1))]])
    R = np.eye(c)[y]-1/c-M
    selection = np.zeros((n,m)); selection[old,np.arange(m)] = 1.
    solved = _solve(H,np.vstack((np.column_stack((R,selection)),np.zeros((1,c+m)))),analysis_work)
    alpha0,b0 = solved[:n,:c],solved[n,:c]
    Jsel,bsel = solved[:n,c:],solved[n,c:]
    P = np.eye(m)-Jsel[old]
    P = .5*(P+P.T)
    pairs = _pairs(y,old,c); rows,truth,other = pairs
    margins = np.array([np.min(np.delete(M[i,y[i]]-M[i],y[i])) for i in old])
    delta = margins[rows]; base = K@alpha0+b0
    s0 = _difference(M[old]+base[old],pairs)-delta
    mu,W = data['multipliers'],data['working_set']
    check(mu.shape==rows.shape and W.dtype.kind in 'iu' and np.array_equal(W,np.unique(W))
          and np.all((W>=0)&(W<len(mu))),'Invalid multipliers/working set')
    V = _scatter(mu,pairs,m,c); fullV = np.zeros((n,c)); fullV[old] = V
    alpha = alpha0+Jsel@V; b = b0+bsel@V
    scores = M+K@alpha+b; slack = _difference(scores[old],pairs)-delta
    for key,value in (('R',R),('pair_rows',rows),('pair_truth',truth),('pair_other',other),
        ('margins',margins),('delta',delta),('P_old',P),('s0',s0),
        ('alpha',alpha),('b',b),('train_scores',scores),('slack',slack)):
        _array_close(data[key],value,'Independent margin primal/dual mismatch: '+key)
    _array_close(A@data['z'],e,'Margin Schur constant equation mismatch')
    close(_scalar(data,'s'),float(np.sum(data['z'])),'Margin Schur scalar mismatch')
    check(np.asarray(data['s']).shape==(),'Margin Schur scalar must remain 0D')
    _factor(data['chol_A'],A,'Margin affine factor')
    QW = _working(P,pairs,W)
    if len(W): _factor(data['chol_working'],QW,'Margin independent active factor')
    else: check(data['chol_working'].shape==(0,0),'Empty active region invented factor')
    _array_close(A@alpha+b-R-fullV,np.zeros_like(R),'Margin canonical stationarity')
    _array_close(alpha.sum(0),np.zeros(c),'Margin free intercept stationarity')
    if np.linalg.norm(M.sum(1))<=128*eps*max(n,c)*max(1.,np.linalg.norm(M)):
        _array_close(alpha.sum(1),np.zeros(n),'Margin class contrast coefficients')
        _array_close(b.sum(),np.asarray(0.),'Margin free intercept class sum')
    scale = max(1.,np.linalg.norm(scores),np.linalg.norm(mu),np.linalg.norm(R),np.linalg.norm(P)*np.linalg.norm(V))
    tolerance = 128*eps*max(n,c,len(mu))*scale
    check(float(np.min(slack))>=-tolerance and float(np.min(mu))>=-tolerance,'Margin primal/dual infeasibility')
    _array_close(mu*slack,np.zeros(len(mu)),'All physical constraint complementarity',scale=scale*scale)
    check(np.count_nonzero(mu[np.setdiff1d(np.arange(len(mu)),W)])==0,'Non-working multiplier was silently active')
    regularizer = float(np.sum(alpha*(K@alpha)))
    primal = float(.5*np.sum((K@alpha+b-R)**2)+.5*regularizer)
    base_obj = float(.5*np.sum((base-R)**2)+.5*np.sum(alpha0*(K@alpha0)))
    qmu = _difference(P@V,pairs)
    dual = float(base_obj-s0@mu-.5*mu@qmu)
    close(primal,dual,'Margin primal/dual gap')
    close(primal-dual,float(mu@slack),'Margin gap identity')
    check(regularizer>=-tolerance,'Negative margin RKHS regularizer')
    if audit is not None:
        final = audit['final_residuals']
        maxabs=lambda value:float(np.max(np.abs(value))) if np.asarray(value).size else 0.
        saved_alpha,saved_b=data['alpha'],data['b']
        for key,value in (('primal_objective',primal),('dual_objective',dual),('regularizer',regularizer),
            ('primal_dual_gap',primal-dual),('minimum_slack',float(np.min(slack))),('minimum_multiplier',float(np.min(mu))),
            ('stationarity',maxabs(A@saved_alpha+saved_b-R-fullV)),('intercept',maxabs(saved_alpha.sum(0))),
            ('complementarity',maxabs(mu*data['slack'])),('gap_identity_error',abs(primal-dual-float(mu@slack)))):
            close(final[key],value,'Margin saved final certificate scalar: '+key)
        for key in ('residual_tolerance','dual_tolerance','gap_tolerance'):
            check(type(final[key]) is float and np.isfinite(final[key]) and final[key]>=0,
                  'Invalid margin declared arithmetic tolerance')
        check(audit['status']=='NUMERIC_KKT_CERTIFIED','Uncertified head presented as success')
    return dict(source=data,H=H,Jsel=Jsel,bsel=bsel,P=P,pairs=pairs,QW=QW,R=R,alpha=alpha,b=b,
        scores=scores,slack=slack,mu=mu,W=W,primal_objective=primal,dual_objective=dual,tolerance=tolerance,
        tight_tolerance=tolerance if audit is None else audit['final_residuals']['residual_tolerance'],
        dual_tolerance=tolerance if audit is None else audit['final_residuals']['dual_tolerance'])


def margin_head_vjp(data,L,G,analysis_work=None,certificate=None):
    """Independent exact active-region Schur KKT adjoint with complete g_b."""
    cert = verify_margin_head(data,analysis_work=analysis_work) if certificate is None else certificate
    check(cert['source'] is data,'Companion certificate belongs to a different head')
    n,c = data['alpha'].shape; old,W = data['old_indices'],cert['W']
    check(L.shape==(len(G),n) and G.shape[1]==c,'Margin companion shape mismatch')
    tolerance = cert['tight_tolerance']
    tight = np.flatnonzero(np.abs(cert['slack'])<=tolerance)
    check(np.array_equal(tight,W) and np.all(cert['mu'][W]>cert['dual_tolerance']),
          'UNSUPPORTED_ACTIVE_JACOBIAN: nonregular active region')
    gb = G.sum(0); rhs = np.vstack((L.T@G,gb[None,:]))
    companion = _solve(cert['H'],rhs,analysis_work)
    T,t = companion[:n],companion[n]
    eta = _solve(cert['QW'],_difference(T[old],cert['pairs'])[W],analysis_work) if len(W) else np.empty(0)
    full_eta = np.zeros(len(cert['mu'])); full_eta[W] = eta
    V = _scatter(full_eta,cert['pairs'],len(old),c)
    W_eta = cert['Jsel']@V
    upstream = -(T+W_eta)@cert['alpha'].T
    return dict(K=.5*(upstream+upstream.T),L=G@cert['alpha'].T,T_G=T,t_G=t,g_b=gb,
        eta=eta,W_eta=W_eta)


def verify_qp_ledger(ledger,operation='forward',n=None,m=None,c=None,allow_failure=False):
    """Sum actual logged calls; verify successful dynamic transition shapes."""
    check(operation in ('forward','adjoint'),'Unknown QP ledger operation')
    factors,solves,events = ledger.get('factorizations',[]),ledger.get('solves',[]),ledger.get('events',[])
    check(ledger['factorization_attempts']==len(factors) and
        ledger['factorizations_completed']==sum(v['status']=='COMPLETED' for v in factors) and
        ledger['condition_estimation_calls']==sum('condition_estimation_info' in v for v in factors),'QP factor ledger mismatch')
    amounts = dict(triangular_calls=len(solves),triangular_rhs_columns=sum(v['rhs_columns'] for v in solves),
        triangular_rhs_elements=sum(v['dimension']*v['rhs_columns'] for v in solves),
        triangular_dense_work_units=sum(v['dimension']**2*v['rhs_columns'] for v in solves))
    for key,value in amounts.items(): check(ledger[key]==value,'QP actual RHS ledger mismatch: '+key)
    for value in solves:
        check(type(value['dimension']) is int and value['dimension']>0 and type(value['rhs_columns']) is int
            and value['rhs_columns']>0 and type(value['transpose']) is bool and
            value['status'] in ('COMPLETED','FAILED','ATTEMPTED'),'Invalid actual triangular record')
    if not allow_failure:
        check(all(v['status']=='COMPLETED' for v in solves+factors),'Successful ledger contains unfinished work')
        check(len(solves)%2==0 and all(solves[j]['transpose'] is False and solves[j+1]['transpose'] is True
            and all(solves[j][key]==solves[j+1][key] for key in ('system','dimension','rhs_columns')) for j in range(0,len(solves),2)),
            'Actual forward/backward triangular pairing mismatch')
    if operation=='adjoint':
        check(not factors and ledger['spectral_checks']==ledger['transitions']==0,'QP adjoint refactorized or ran optimizer')
    elif not allow_failure:
        check(n is not None and m is not None and c is not None,'Missing QP physical dimensions')
        check(ledger['transitions']==len(events) and 1<=len(events)<=ledger['max_transitions']
            and ledger['spectral_checks']==1 and ledger['spectral_cubic_dimension_units']==n**3
            and ledger['compact_snapshot_rebuilds']==len(events)
            and ledger['compact_snapshot_dense_work_units']==len(events)*m*m*c
            and ledger['full_constraint_scans']==len(events)+1,'QP full physical scans/spectral work mismatch')
        expected_factors = [('affine',n)]+[('working',len(event['working'])) for event in events if event['working']]
        check([(v['system'],v['dimension']) for v in factors]==expected_factors,'QP actual transition factors mismatch')
        expected = [('affine_constant',n,1),('affine_base',n,c),('old_response_columns',n,m)]
        independence = 0
        for index,event in enumerate(events):
            check(event['transition']==index and event['working']==sorted(set(event['working'])),'QP transition index/working mismatch')
            a = len(event['working'])
            if a: expected.append(('working_multiplier',a,1))
            if event.get('blocker') is not None:
                independence += 1
                if a: expected.append(('working_independence',a,1))
        expected.append(('affine_final_readback',n,c))
        check([(v['system'],v['dimension'],v['rhs_columns']) for v in solves[::2]]==expected
              and ledger['independence_checks']==independence,'QP transition/RHS system mismatch')
    peak = max((16*v['dimension']*v['rhs_columns'] for v in solves),default=0)
    check(ledger['peak_explicit_solve_temporary_bytes']==peak,'QP actual solve temporary MAX mismatch')
    if factors:
        order = n if n is not None else next((v['dimension'] for v in factors if v['system']=='affine'),0)
        factor_peak = max((8*v['dimension']**2*(2 if v['status']=='COMPLETED' else 1)+(16*order*order if v['system']=='working' else 0) for v in factors),default=0)
        check(ledger['peak_factor_buffer_bytes']==factor_peak,'QP actual factor MAX mismatch')
    elif operation=='adjoint': check(ledger['peak_factor_buffer_bytes']==0,'Adjoint invented new factor peak')
    check(ledger['peak_factor_buffer_bytes']<=ledger['max_factor_buffer_bytes'],'QP caller factor bound exceeded')
    return {key:ledger.get(key,0) for key in QP_WORK_KEYS}


def verify_failed_fit(audit,arrays,analysis_work=None):
    """Verify available failed-work evidence; never certify an optimal head/run."""
    check(audit['status']=='TECHNICAL_FAILURE' and audit.get('failure_code'),
          'Failure certificate requires explicit technical status/code')
    partial = audit.get('qp_failure_audit') or {}
    if 'forward' in partial:
        ledger=partial['forward']
        verify_qp_ledger(ledger,'forward',ledger['n'],ledger['m'],ledger['classes'])
        operation='forward'
    else:
        ledger=partial
        check(all(key in ledger for key in QP_WORK_KEYS),'Unknown failed solver work cannot be zero-filled')
        operation='adjoint' if ledger.get('operation')=='REGULAR_VJP' else 'forward'
        verify_qp_ledger(ledger,operation,ledger.get('n'),ledger.get('m'),ledger.get('classes'),allow_failure=True)
    for key in QP_WORK_KEYS:
        check('margin_qp_'+operation+'_'+key in audit and audit['margin_qp_'+operation+'_'+key]>=ledger.get(key,0),
              'Last failed work disappeared from cumulative stage: '+key)
    result=dict(status='FAILED_PARTIAL_EVIDENCE_VERIFIED',failure_code=audit['failure_code'],
        optimality_certified=False,regular_jacobian_certified=False,complete_run=False,
        cumulative_work_verified=False,known_last_solver_work={key:ledger.get(key,0) for key in QP_WORK_KEYS},
        known_last_solver_peaks={key:ledger.get(key,0) for key in ('peak_factor_buffer_bytes','peak_explicit_solve_temporary_bytes')},
        reason='Partial last-call ledger is checked; prior completed stages remain separate and no missing work is invented')
    if any(not np.isfinite(np.asarray(value)).all() for value in arrays.values()):
        result['numeric_snapshot_status']='NONFINITE_TECHNICAL_EVIDENCE_PRESERVED_NOT_CERTIFIED'
        return result
    snapshot={key.removeprefix('qp_failure_'):value for key,value in arrays.items() if key.startswith('qp_failure_')}
    if {'K','M','labels','old_indices','rho','V','P_old','slack','margins'}<=set(snapshot):
        K,M,y,old=(snapshot[key] for key in ('K','M','labels','old_indices'))
        n,c=M.shape; R=np.eye(c)[y]-1/c-M
        H=np.block([[K+np.eye(n),np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
        base=_solve(H,np.vstack((R,np.zeros((1,c)))),analysis_work)
        base_scores=K@base[:n]+base[n]
        pairs=_pairs(y,old,c)
        margin=np.array([np.min(np.delete(M[i,y[i]]-M[i],y[i])) for i in old])
        current=float(snapshot['rho'])*base_scores[old]+snapshot['P_old']@snapshot['V']
        _array_close(snapshot['margins'],margin,'Failed snapshot original margins')
        _array_close(snapshot['slack'],_difference(M[old]+current,pairs)-margin[pairs[0]],
                     'Failed snapshot current rho/V/slack consistency')
        result['numeric_snapshot_status']='CURRENT_COMPACT_STATE_RECONSTRUCTED_NO_OPTIMALITY_CLAIM'
    else: result['numeric_snapshot_status']='AVAILABLE_ARRAYS_PRESERVED_NO_COMPLETE_PRIMAL_SNAPSHOT'
    return result


def _head_work(data,audit):
    n,h,c = len(data['train_labels']),len(data['held_labels']),data['Y'].shape[1]
    positive = int(_scalar(data,'gamma') is not None); b = audit['mode']=='B'
    if b:
        expected = dict(factorization_count=positive,completed_factorization_count=positive,
            head_triangular_solve_count=2*positive,head_triangular_rhs_count=2*(c+1)*positive,
            head_triangular_rhs_element_count=2*n*(c+1)*positive,head_triangular_dense_work_unit_count=2*n*n*(c+1)*positive)
        expected.update({'margin_qp_forward_'+key:0 for key in QP_WORK_KEYS})
    else:
        qpwork = verify_qp_ledger(audit['margin_qp_audit'],'forward',n,len(data['old_indices']),c)
        expected = {'margin_qp_forward_'+key:value for key,value in qpwork.items()}
        expected.update(factorization_count=qpwork['factorization_attempts'],completed_factorization_count=qpwork['factorizations_completed'],
            head_triangular_solve_count=qpwork['triangular_calls'],head_triangular_rhs_count=qpwork['triangular_rhs_columns'],
            head_triangular_rhs_element_count=qpwork['triangular_rhs_elements'],head_triangular_dense_work_unit_count=qpwork['triangular_dense_work_units'])
        for key in PEAK_COUNTERS:
            check(audit.get(key,0)==audit['margin_qp_audit'][key.removeprefix('margin_qp_')],
                  'Saved head QP peak alias mismatch: '+key)
    identity = not np.any(data['U']) or _scalar(data,'tau')==0; m = len(data['old_indices'])
    expected.update(head_fit_count=1,ajlr_forward_evaluation_count=1,intercept_fit_count=1,intercept_addition_count=(n+h)*c,
        raw_distance_evaluation_count=(1+int(h>0))*int(not identity),raw_distance_pair_count=(n*(n-1)//2+h*n)*int(not identity),
        reference_distance_evaluation_count=(1+int(h>0))*int(not identity),
        reference_distance_pair_count=(m*(m-1)//2+m*(n-m)+h*m)*int(not identity),
        kernel_evaluation_count=(1+int(h>0))*positive,kernel_pair_count=(n*n+h*n)*positive,adapter_physical_evaluation_count=n+h)
    for key,value in expected.items(): check(type(audit[key]) is int and audit[key]==value,'Actual head work mismatch: '+key)


def verify_head(ref,audit,resolver,prior=None):
    data = resolver(ref); ids,held,classes = audit['training_physical_ids'],audit['held_physical_ids'],audit['classes']
    n,h,c = len(ids),len(held),len(classes); y,yh = data['train_labels'],data['held_labels']
    check(audit['schema']==SCHEMA and audit['method']==METHOD and audit['mode'] in ('B','C_seq'),'Head schema/mode mismatch')
    check(y.dtype.kind in 'iu' and yh.dtype.kind in 'iu' and y.shape==(n,) and yh.shape==(h,)
        and np.all((y>=0)&(y<c)) and np.all((yh>=0)&(yh<c)),'Physical labels mismatch')
    old = audit['old_classes']; oi = np.array([i for i,v in enumerate(y) if classes[int(v)] in old],dtype=np.int64)
    ni = np.array([i for i in range(n) if i not in set(oi)],dtype=np.int64)
    check(len(set(ids+held))==n+h and audit['old_reference_physical_ids']==[ids[i] for i in oi],'Physical old reference mismatch')
    for key,expected in (('old_indices',oi),('new_indices',ni)):
        _array_close(data[key],expected,'Physical old/new rows mismatch: '+key)
    q = np.zeros(n); q[oi]=1/len(oi); _array_close(data['q'],q,'B old-reference measure changed')
    ob,oa,hb,ha = [data[key] for key in ('original_train_b','original_train_a','original_held_b','original_held_a')]
    ub,ua = adapted_blocks(_raw(ob,oa),data['U']); uhb,uha = adapted_blocks(_raw(hb,ha),data['U'])
    for key,expected in (('adapted_train_b',ub),('adapted_train_a',ua),('adapted_held_b',uhb),('adapted_held_a',uha)):
        _array_close(data[key],expected,'Independent adapter mapping: '+key)
    d0,dh0 = _distances(ob,oa),_distances(hb,ha,ob,oa)
    tau,gamma = (_scalar(data,key) for key in ('tau','gamma'))
    if audit['mode']=='B':
        s0 = _scalar(data,'s0')
    else:
        reference_s0 = np.asarray(data['reference_s0'])
        check(reference_s0.shape==() and reference_s0.size==1 and np.isfinite(reference_s0).all(),
              'C reference_s0 must be one finite 0D scalar')
        s0 = float(reference_s0)
    d = d0 if tau==0 or not np.any(data['U']) else .5*(d0+_distances(ub,ua))
    dh = dh0 if tau==0 or not np.any(data['U']) else .5*(dh0+_distances(uhb,uha,ub,ua))
    for key,expected in (('original_distance',d0),('original_cross_distance',dh0),('distance',d),('cross_distance',dh)):
        _array_close(data[key],expected,'Independent complete distance: '+key)
    od = d0[np.ix_(oi,oi)]; oy=y[oi]; expected_s0=float(np.sum(od[np.triu_indices(len(oi),1)])/len(oi))
    expected_tau = None if len(set(oy.tolist()))==1 else float(np.median(np.min(np.where(oy[:,None]!=oy[None,:],od,np.inf),axis=1)))
    close(s0,expected_s0,'Original old R0 trace'); close(tau,expected_tau,'Original old R0 bandwidth')
    expected_gamma = None if tau is None or s0==0 else s0/float(-2*np.sum(_radial(od,tau,1.,True)[np.triu_indices(len(oi),1)])/len(oi))
    close(gamma,expected_gamma,'Original old R0 gamma')
    radial,cross = _radial(d,tau,gamma),_radial(dh,tau,gamma)
    _array_close(data['raw_train'],radial,'Raw complete train kernel'); _array_close(data['raw_cross'],cross,'Raw complete held kernel')
    Y=np.eye(c)[y]-1/c; _array_close(data['Y'],Y,'Centered row target')
    M,Mh=data['M_train'],data['M_held']
    _array_close(M.sum(1),np.zeros(n),'Frozen actual B train class contrast')
    _array_close(Mh.sum(1),np.zeros(h),'Frozen actual B held class contrast')
    if prior is None:
        check(audit['mode']=='B','C missing actual B')
        _array_close(M,np.zeros((n,c)),'B nonzero prior'); _array_close(Mh,np.zeros((h,c)),'B nonzero held prior')
    else:
        pdata,pclasses=prior; check(audit['mode']=='C_seq' and pclasses==old,'Actual B class binding')
        for key,expected in (('original_train_b',ob[oi]),('original_train_a',oa[oi])):
            _array_close(pdata[key],expected,'Actual B old raw geometry mismatch')
        for raw,expected in ((_raw(ob,oa),M),(_raw(hb,ha),Mh)):
            values=np.zeros(expected.shape); values[:,[classes.index(v) for v in pclasses]]=head_score(pdata,raw)
            _array_close(expected,values,'Frozen actual B function mapping')
    target=Y-M; _array_close(data['residual_target'],target,'Residual target recentered')
    if audit['mode']=='B':
        augmented=dict(data,E=target,raw_train_minus_one=_radial(d,tau,gamma,True),
            raw_cross_minus_one=_radial(dh,tau,gamma,True),actual_trace=np.asarray(float(np.trace(data['K']))))
        affine_analysis.verify_head(ref,audit,lambda unused:augmented)
    else:
        _array_close(data['K'],np.zeros_like(d) if gamma is None else gamma*radial,'C full physical raw Gram')
        _array_close(data['L'],np.zeros_like(dh) if gamma is None else gamma*cross,'C full physical raw cross')
        _array_close(data['M'],M,'QP frozen actual B table'); _array_close(data['labels'],y,'QP physical labels')
        verify_margin_head(data,audit['margin_qp_audit'],getattr(resolver,'analysis_work',None))
        _array_close(data['scores'],Mh+data['L']@data['alpha']+data['b'],'C complete free-intercept held scores')
        close(audit['actual_kernel_trace'],float(np.trace(data['K'])),'C raw trace audit')
        close(audit['normal_equation_residual'],audit['margin_qp_audit']['final_residuals']['stationarity'],'C stationarity alias')
        close(audit['head_training_loss_data'],float(.5*np.sum((data['train_scores']-Y)**2)),'C data loss audit')
        close(audit['head_training_loss_ridge'],float(.5*np.sum(data['alpha']*(data['K']@data['alpha']))),'C RKHS loss audit')
        close(audit['head_training_loss_total'],audit['head_training_loss_data']+audit['head_training_loss_ridge'],'C complete head loss audit')
    _head_work(data,audit)
    return data


def verify_preparation(prep, entry, labels, old, resolver):
    b = prep['state'] == 'B'; ids = entry['b_training_ids' if b else 'c_training_ids']; classes = entry['b_classes' if b else 'c_classes']
    k = entry['train_k']; folds = 0 if k == 1 else min(k, 3)
    check(prep['schema'] == SCHEMA and prep['method'] == METHOD and prep['training_physical_ids'] == ids
          and prep['classes'] == classes and prep['old_classes'] == old and prep['train_k'] == k
          and prep['train_physical_count'] == len(ids) and prep['ajlr_preparation_count'] == 1
          and prep['inherited_state'] is (not b) and prep['inherited_adapter_from'] == (None if b else 'B_MARGIN')
          and prep['source_inputs'] is False and prep['query_fit'] is False, 'Preparation physical/current-B binding mismatch')
    check(type(prep['qp_max_transitions']) is int and prep['qp_max_transitions']>0 and
        type(prep['qp_max_factor_buffer_bytes']) is int and prep['qp_max_factor_buffer_bytes']>0,'Missing actual QP resources')
    check(prep['actual_parameters']==dict(PROBE_CONFIG,qp_max_transitions=prep['qp_max_transitions'],
        qp_max_factor_buffer_bytes=prep['qp_max_factor_buffer_bytes']),'Preparation actual QP config mismatch')
    affine_analysis.verify_latent_coordinates(prep, resolver)
    check(len(prep['inner_folds']) == folds and len(prep['prior_folds']) == (0 if b else folds), 'Inner/prior fold coverage mismatch')
    assignment = {pid: j % folds for cls in classes for j, pid in enumerate(sorted(pid for pid in ids if labels[pid] == cls))} if folds else {}
    prior_arrays = []; seen = set(); original_pairs = 0
    for j, fold in enumerate(prep['inner_folds']):
        held = [pid for pid in ids if assignment[pid] == j]; train = [pid for pid in ids if assignment[pid] != j]
        check(fold['inner_fold'] == j and fold['training_physical_ids'] == train and fold['held_physical_ids'] == held
              and fold['classes'] == classes and fold['old_classes'] == old
              and fold['old_reference_physical_ids'] == [pid for pid in train if labels[pid] in old]
              and fold['all_head_statistics_from_old_inner_train_only'] is True
              and not set(train + held).intersection(entry['c_ids']), 'Inner/outer physical isolation mismatch')
        check(not seen.intersection(held), 'Duplicate inner held physical ID'); seen.update(held)
        n, h = len(train), len(held); original_pairs += n * (n - 1) // 2 + h * n
        if b: check(fold['prior_source'] == 'ZERO' and fold['prior_ref'] is None, 'B invented prior')
        else:
            prior = prep['prior_folds'][j]
            check(prior['training_physical_ids'] == [pid for pid in train if labels[pid] in old]
                  and prior['held_physical_ids'] == [pid for pid in held if labels[pid] in old]
                  and prior['classes'] == old and fold['prior_source'] == 'FROZEN_CURRENT_ACTUAL_B'
                  and fold['prior_ref'] == prior['head_state_ref'], 'Inner actual-B prior physical binding mismatch')
            pdata = verify_head(prior['head_state_ref'], prior['final_fit'], resolver)
            check(np.array_equal(pdata['U'], resolver(prep['prepared_state_ref'])['anchor_U']), 'Inner prior mapping is not frozen actual B U')
            prior_arrays.append(pdata)
            pn, ph = len(prior['training_physical_ids']), len(prior['held_physical_ids'])
            original_pairs += pn * (pn - 1) // 2 + ph * pn
    check(not folds or seen == set(ids), 'Incomplete physical inner-held coverage')
    original_pairs += len(ids) * (len(ids) - 1) // 2
    check(prep['prepared_distance_evaluation_count'] == (2 * folds + 1 if b else 4 * folds + 1)
          and prep['original_distance_pair_count'] == original_pairs, 'Actual original preparation distance work mismatch')
    factors = sum(int(_scalar(value, 'gamma') is not None) for value in prior_arrays)
    check(prep['prior_head_fit_count'] == len(prior_arrays) and prep['prior_factorization_count'] == factors
          and prep['prior_triangular_solve_count'] == 2 * factors
          and prep['prior_score_evaluation_count'] == (0 if b else 2 * folds + 1)
          and prep['prior_score_physical_count'] == (0 if b else (folds + 1) * len(ids)), 'Actual prior preparation work mismatch')
    for suffix, power in (('rhs_count', 0), ('rhs_element_count', 1), ('dense_work_unit_count', 2)):
        expected = sum(2 * len(value['alpha']) ** power * (len(old) + 1) for value in prior_arrays if _scalar(value, 'gamma') is not None)
        check(prep['prior_triangular_' + suffix] == expected, 'Prior C+1 RHS work mismatch: ' + suffix)
    geometry=dict.fromkeys(('reference_distance_evaluation_count','reference_distance_pair_count',
        'raw_distance_evaluation_count','raw_distance_pair_count','kernel_evaluation_count','kernel_pair_count',
        'adapter_physical_evaluation_count'),0)
    # Original old nuisance distances are real preparation work. Current C
    # reuses each actual prior's scale; no second nuisance solve is invented.
    old_counts=[len(f['old_reference_physical_ids']) for f in prep['inner_folds']]
    nuisance_counts=old_counts+([len(ids)] if b else [])
    geometry['reference_distance_evaluation_count']=len(nuisance_counts)
    geometry['reference_distance_pair_count']=sum(v*(v-1)//2 for v in nuisance_counts)
    intercept_additions=0
    if not b:
        priors=prior_arrays+[resolver(prep['final_problem']['prior_ref'])]
        for j,pdata in enumerate(priors):
            if j<folds:
                paudit=prep['prior_folds'][j]['final_fit']
                for key in geometry:geometry[key]+=paudit[key]
                intercept_additions+=paudit['intercept_addition_count']
            pn=len(pdata['train_labels']);gamma=_scalar(pdata,'gamma');tau=_scalar(pdata,'tau')
            calls=0 if gamma is None else len(ids)*(1+int(tau!=0 and bool(np.any(pdata['U']))))
            geometry['raw_distance_evaluation_count']+=calls
            geometry['raw_distance_pair_count']+=calls*pn
            geometry['reference_distance_evaluation_count']+=calls
            geometry['reference_distance_pair_count']+=calls*len(pdata['old_indices'])
            geometry['kernel_evaluation_count']+=len(ids)*int(gamma is not None)
            geometry['kernel_pair_count']+=len(ids)*pn*int(gamma is not None)
            geometry['adapter_physical_evaluation_count']+=len(ids)*int(gamma is not None and tau!=0)
            intercept_additions+=len(ids)*len(old)
    for key,value in geometry.items():check(prep[key]==value,'Preparation actual geometry/prior score cost: '+key)
    check(prep['prior_intercept_fit_count']==len(prior_arrays) and prep['prior_intercept_addition_count']==intercept_additions
          and prep['dictionary_physical_evaluation_count']==len(ids),'Preparation real dictionary/intercept cost')
    reason = 'PHYSICAL_K1' if k == 1 else 'ZERO_DICTIONARY_RANK' if prep['latent_rank'] == 0 else \
        'NO_OLD_KERNEL_INFORMATION' if prep['final_problem']['trace_scale'] is None else \
        'ALL_INNER_GEOMETRY_DEGENERATE' if not any(f['trace_scale'] is not None and f['bandwidth_tau'] is not None and f['bandwidth_tau'] > 0 for f in prep['inner_folds']) else None
    check(prep['no_information_reason'] == reason and prep['no_information'] is (reason is not None), 'No-information/no-update reason mismatch')


def _kernel_vjp(data,G,fold,enabled,analysis_work=None):
    """General affine/active-KKT solve and independently traversed endpoints."""
    n,h,c = len(data['train_labels']),len(data['held_labels']),data['Y'].shape[1]
    tau,gamma = _scalar(data,'tau'),_scalar(data,'gamma')
    regular_information = gamma is not None and gamma>0 and tau is not None and tau>0 and h>0
    is_b = 'intercept' in data
    if not enabled or not regular_information:
        for key in ('ce_adjoint_solve_count',)+tuple('derivative_'+v for v in SUFFIXES[1:])+tuple('margin_qp_adjoint_'+v for v in QP_WORK_KEYS):
            check(fold[key]==0,'Invented unavailable/cached adjoint cost: '+key)
        if not regular_information or 'adjoint_G' not in data:
            return np.zeros((736,8))
        # Immutable cached arrays can retain the previously computed companion.
        # Independently check it again, without assigning it new training work.
    else:
        check(fold['ce_adjoint_solve_count']==1,'Missing actual CE companion call')
    if is_b:
        H = np.block([[data['K']+np.eye(n),np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
        rhs = np.vstack((data['L'].T@G,G.sum(0)[None,:]))
        solution = _solve(H,rhs,analysis_work); T,t = solution[:n],solution[n]
        independent = dict(T_G=T,t_G=t,g_b=G.sum(0),rhs=rhs[:n],
            K=-.5*(T@data['alpha'].T+data['alpha']@T.T),L=G@data['alpha'].T)
        barR,barQ = center_kernel_vjp(independent['K'],independent['L'],data['q'],gamma)
        if enabled:
            expected = dict(derivative_triangular_solve_count=2,derivative_triangular_rhs_count=2*c,
                derivative_triangular_rhs_element_count=2*n*c,derivative_triangular_dense_work_unit_count=2*n*n*c)
            for key,value in expected.items(): check(fold[key]==value,'Actual affine complete companion cost: '+key)
    else:
        cert = verify_margin_head(data,fold['margin_qp_audit'],analysis_work)
        independent = margin_head_vjp(data,data['L'],G,analysis_work,cert)
        barR,barQ = gamma*independent['K'],gamma*independent['L']
        if enabled:
            ledger=fold['margin_qp_adjoint_audit']; work=verify_qp_ledger(ledger,'adjoint')
            for key,value in work.items(): check(fold['margin_qp_adjoint_'+key]==value,'Actual complete QP companion cost: '+key)
            for key,rawkey in (('solve_count','triangular_calls'),('rhs_count','triangular_rhs_columns'),
                ('rhs_element_count','triangular_rhs_elements'),('dense_work_unit_count','triangular_dense_work_units')):
                check(fold['derivative_triangular_'+key]==work[rawkey],'QP derivative RHS alias mismatch')
            expected=[('adjoint_affine',n,c)]
            if len(cert['W']): expected += [('adjoint_working',len(cert['W']),1),('adjoint_multiplier_response',n,c)]
            check([(v['system'],v['dimension'],v['rhs_columns']) for v in ledger['solves'][::2]]==expected,
                'QP complete multiplier/free-intercept system cost mismatch')
    _array_close(data['adjoint_G'],G,'Actual held CE upstream mismatch')
    for key,value in independent.items(): _array_close(data['adjoint_'+key],value,'Independent full KKT companion: '+key)
    dd,dc = -.5*barR*data['raw_train']/tau,-.5*barQ*data['raw_cross']/tau
    for key,value in (('raw_train_upstream',barR),('raw_cross_upstream',barQ),
        ('adapted_distance_upstream',dd),('adapted_cross_distance_upstream',dc)):
        _array_close(data['adjoint_'+key],value,'All kernel endpoints upstream: '+key)
    b,a,hb,ha=[data[key] for key in ('adapted_train_b','adapted_train_a','adapted_held_b','adapted_held_a')]
    gb,ga,rb,ra=affine_analysis._distance_reverse(b,a,b,a,dd)
    hgb,hga,tgb,tga=affine_analysis._distance_reverse(hb,ha,b,a,dc)
    gU=affine_analysis._adapter_reverse(data['original_train_b'],data['original_train_a'],data['U'],
        np.column_stack((gb+rb+tgb,ga+ra+tga)))
    gU+=affine_analysis._adapter_reverse(data['original_held_b'],data['original_held_a'],data['U'],np.column_stack((hgb,hga)))
    _array_close(data['adjoint_g_U'],gU,'Independent complete adapter gradient',scale=np.linalg.norm(gU))
    return gU if enabled else np.zeros((736,8))


def verify_objective(value, Z, U, prep, resolver, aggregate_ref):
    folds = value['inner_folds']; c = len(prep['classes']); sums = np.zeros(c); counts = np.zeros(c, dtype=np.int64); heads = []
    check(len(folds) == len(prep['inner_folds']) and value['temperature'] == 1., 'Objective inner fold/temperature mismatch')
    aggregate = resolver(aggregate_ref); gradient = 'g_Z' in aggregate
    for j, (fold, physical) in enumerate(zip(folds, prep['inner_folds'])):
        check(all(fold[key] == physical[key] for key in ('classes', 'training_physical_ids', 'held_physical_ids', 'old_reference_physical_ids', 'bandwidth_tau', 'trace_scale', 'prior_ref')),
              'Objective physical/current-B binding mismatch')
        prior = None if prep['state'] == 'B' else (resolver(prep['prior_folds'][j]['head_state_ref']), prep['old_classes'])
        data = verify_head(fold['head_state_ref'], fold, resolver, prior); heads.append(data)
        check(np.array_equal(data['U'], U), 'Objective head mapping changed')
        f, y = data['scores'], data['held_labels']; top = f.max(1)
        ce = np.log(np.exp(f - top[:, None]).sum(1)) + top - f[np.arange(len(y)), y]
        fs, fc = np.bincount(y, weights=ce, minlength=c), np.bincount(y, minlength=c)
        _array_close(fold['held_ce_sums'], fs, 'Physical fold CE sum mismatch')
        check(fold['held_ce_counts'] == fc.tolist() and fold['held_correct_count'] == int(np.sum(np.argmax(f, axis=1) == y)), 'Fold CE counts/accuracy mismatch')
        sums += fs; counts += fc
    check(np.all(counts == prep['train_k']), 'CE did not pool every physical inner-held row exactly once')
    means = sums / counts; risk = float(np.linalg.norm(means)) / math.sqrt(c)
    for key, expected in (('class_ce_sums', sums), ('class_ce_counts', counts), ('class_ce_means', means)):
        _array_close(value[key], expected, 'Pooled RMS class CE mismatch: ' + key)
        _array_close(aggregate[key], expected, 'Aggregate class CE coordinate mismatch: ' + key)
    for key in ('loss_ce', 'loss_task', 'RMSCE', 'loss_total'): close(value[key], risk, 'CE-only objective mismatch: ' + key)
    close(value['loss_proximal'], 0., 'CE-only objective added proximal loss')
    _array_close(aggregate['prox'], np.asarray(0.), 'CE-only aggregate added proximal loss')
    _array_close(aggregate['RMSCE'], np.asarray(risk), 'Aggregate RMSCE mismatch')
    check(value['coordinate_ball_radius'] == .5 and value['coordinate_ball_feasible'] is True and np.linalg.norm(Z) <= .5 + 128 * np.finfo(float).eps,
          'Coordinate ball contract mismatch')
    reused = value['forward_cache_reused']; check(type(reused) is bool and value['backward_evaluation_count'] == int(gradient), 'Cached objective/gradient flag mismatch')
    geometry=('raw_distance_evaluation_count','raw_distance_pair_count','reference_distance_evaluation_count',
        'reference_distance_pair_count','kernel_evaluation_count','kernel_pair_count','adapter_physical_evaluation_count','ajlr_forward_evaluation_count')
    for key in KERNEL_COUNTERS + geometry + tuple('head_' + s for s in SUFFIXES[1:]) + ('intercept_fit_count', 'intercept_addition_count'):
        expected = sum(f[key] for f in folds) if 'adjoint' in key else (0 if reused else sum(f[key] for f in folds))
        check(value[key] == expected, 'Objective actual work/cached forward mismatch: ' + key)
    for key in ('ce_adjoint_solve_count',) + tuple('derivative_' + s for s in SUFFIXES[1:]):
        check(value[key] == sum(f[key] for f in folds), 'Objective complete adjoint count mismatch: ' + key)
    for key in PEAK_COUNTERS:
        expected=max((f.get(key,0) for f in folds),default=0)
        # Gradient audits retain associated forward-factor high-water marks;
        # this is a MAX of owned work, never a repeated factor SUM charge.
        if reused and not gradient: expected=0
        check(value.get(key,0)==expected,'Objective actual peak MAX/cache mismatch: '+key)
    check(value['inner_head_fit_count'] == (0 if reused else len(folds))
          and value['inner_factorization_count'] == (0 if reused else sum(f['factorization_count'] for f in folds))
          and value['inner_objective_evaluation_count'] == int(not reused), 'Objective actual head/factor/cache mismatch')
    if gradient:
        gU = np.zeros((736, 8))
        for data, fold in zip(heads, folds):
            f, y = data['scores'], data['held_labels']; ex = np.exp(f - f.max(1)[:, None]); G = ex / ex.sum(1)[:, None]
            G[np.arange(len(y)), y] -= 1
            if risk: G *= (means[y] / (c * risk * counts[y]))[:, None]
            else: G.fill(0.)
            gU += _kernel_vjp(data, G, fold, risk > 0, getattr(resolver, 'analysis_work', None))
        W = resolver(prep['prepared_state_ref'])['W']
        _array_close(aggregate['g_Z'], gU @ W, 'Independent CE-only coordinate gradient mismatch', scale=np.linalg.norm(gU) * np.linalg.norm(W))
    else:
        for data,fold in zip(heads,folds):
            f,y=data['scores'],data['held_labels']; ex=np.exp(f-f.max(1)[:,None]); G=ex/ex.sum(1)[:,None]
            G[np.arange(len(y)),y]-=1
            if risk: G*=(means[y]/(c*risk*counts[y]))[:,None]
            else: G.fill(0.)
            _kernel_vjp(data,G,fold,False,getattr(resolver,'analysis_work',None))
    return risk


def _dictionary(raw):
    from scipy.special import erf
    full = np.column_stack(tuple(raw[key] for key in BRANCHES)); unit = np.zeros_like(full)
    for sl in (slice(0, 160), slice(160, 256), slice(256, 416), slice(416, 576), slice(576, 736)):
        for i, row in enumerate(full):
            norm = float(np.linalg.norm(row[sl]))
            if norm: unit[i, sl] = row[sl] / norm
    z = np.asarray([np.sum(dct_initial() * (row / math.sqrt(5))[None, :], axis=1) for row in unit])
    return .5 * z * (1 + erf(z / math.sqrt(2)))


def verify_candidate(stage, prep, entry, b_stage, resolver):
    coordinates = resolver(prep['prepared_state_ref']); W, H = coordinates['W'], coordinates['H']; rank = prep['latent_rank']
    anchor = np.zeros((736, 8)) if stage['mode'] == 'B' else resolver(b_stage['final_state_ref'])['U']
    check(stage['status'] == 'COMPLETED' and stage['schema'] == SCHEMA and stage['method'] == METHOD and stage['config'] == dict(PROBE_CONFIG,qp_max_transitions=prep['qp_max_transitions'],qp_max_factor_buffer_bytes=prep['qp_max_factor_buffer_bytes'])
          and stage['preparation']['prepared_state_ref'] == prep['prepared_state_ref']
          and stage['preparation']['inner_folds'] == prep['inner_folds'] and stage['preparation']['prior_folds'] == prep['prior_folds']
          and stage['no_information'] is prep['no_information'], 'Candidate preparation/schema/frozen structure mismatch')
    for key in ('run_id', 'row_id', 'split_id', 'scope', 'fold', 'parent_k', 'train_k'):
        check(stage['preparation'][key] == prep[key], 'Candidate current-B lineage mismatch: ' + key)
    def state(ref):
        data = resolver(ref); Z, U = data['Z'], data['U']
        check(Z.shape == (736, rank) and U.shape == (736, 8) and np.array_equal(data['W'], W)
              and np.array_equal(data['anchor_U'], anchor), 'Coordinate/actual-B anchor binding mismatch')
        affine_analysis.verify_coordinate_map(Z, U, anchor, W)
        check(np.linalg.norm(Z) <= .5 + 128 * np.finfo(float).eps, 'Coordinate ball exceeded')
        return data, Z, U
    initial, Z, U = state(stage['initialization_state_ref'])
    check(np.array_equal(Z, np.zeros((736, rank))) and np.array_equal(U, anchor), 'Exact B-to-C initialization mismatch')
    steps, gradients, trials = stage['steps'], stage['gradients'], stage['trials']; objectives = []
    check(stage['optimizer_steps'] == stage['accepted_trial_count'] == len(steps) <= 4
          and stage['optimizer_iterations'] == stage['backward_evaluation_count'] == len(gradients) <= 4
          and stage['trial_count'] == stage['trial_attempt_count'] == len(trials) <= 48
          and stage['rejected_trial_count'] == len(trials) - len(steps), 'Fixed 4x12 actual solver counts mismatch')
    if prep['no_information']:
        check(not steps and not gradients and not trials and stage.get('initial_objective') is None and stage.get('final_objective') is None
              and stage['stop_reason'] == prep['no_information_reason'], 'Invented no-information adapter training')
        current = None
    else:
        current = verify_objective(stage['initial_objective'], Z, U, prep, resolver, stage['initialization_state_ref'])
        objectives.append(stage['initial_objective'])
    used = 0; accepted = []; path_length = 0.
    for iteration, event in enumerate(gradients, 1):
        data, gz, gu = state(event['state_ref']); g, direction = data['g_Z'], data['d_Z']; norm = float(np.linalg.norm(g))
        check(event['iteration'] == iteration and np.array_equal(gz, Z) and np.array_equal(gu, U), 'Gradient cache changed current coordinates')
        _array_close(direction, -g / norm if norm else np.zeros_like(g), 'Normalized negative CE gradient mismatch')
        close(event['gradient_norm'], norm, 'Actual gradient norm mismatch'); close(event['direction_norm'], float(np.linalg.norm(direction)), 'Direction norm mismatch')
        loss = verify_objective(event['objective'], Z, U, prep, resolver, event['state_ref']); close(loss, current, 'Accepted gradient cache changed RMSCE')
        check(event['objective']['forward_cache_reused'] is True, 'Gradient refitted accepted heads'); objectives.append(event['objective'])
        group = [trial for trial in trials if trial['iteration'] == iteration]; check(len(group) <= 12 and (norm > 0 or not group), 'Trial after zero gradient/budget exceeded')
        winner = None
        for j, trial in enumerate(group, 1):
            check(winner is None and trial['trial'] == j and trial['step_size'] == .125 * .5 ** (j - 1)
                  and trial['gradient_state_ref'] == event['state_ref'], 'Fixed first-acceptable trial ordering mismatch')
            td, tz, tu = state(trial['state_ref']); proposal = Z + trial['step_size'] * direction; pn = float(np.linalg.norm(proposal))
            expected = proposal if pn <= .5 else proposal * (.5 / pn); delta = tz - Z
            check(np.array_equal(tz, expected) and np.array_equal(td['delta_Z'], delta) and np.any(delta), 'Projected actual delta mismatch')
            after = verify_objective(trial['objective'], tz, tu, prep, resolver, trial['state_ref']); objectives.append(trial['objective'])
            dot = float(np.sum(g * delta)); bound = current + 1e-4 * dot
            tol = 128 * np.finfo(float).eps * max(1., abs(current), abs(after), abs(bound))
            armijo, nonincrease = bool(after <= bound + tol), bool(after <= current + tol)
            check(trial['armijo_pass'] is armijo and trial['objective_nonincrease_pass'] is nonincrease
                  and trial['accepted'] is (armijo and nonincrease), 'Actual-delta CE-only Armijo mismatch')
            for key, expected_scalar in (('comparison_tolerance', tol), ('gradient_dot_delta', dot), ('update_norm', float(np.linalg.norm(delta))), ('loss_before', current), ('loss_after', after)):
                close(trial[key], expected_scalar, 'Trial scalar mismatch: ' + key)
            used += 1
            if trial['accepted']: winner = trial
        if winner is not None:
            step = steps[len(accepted)]
            check(step['iteration'] == iteration and step['trial'] == winner['trial'] and step['state_ref'] == winner['state_ref']
                  and step['objective'] == winner['objective'], 'Accepted step/trial binding mismatch')
            accepted.append(step); path_length += winner['update_norm']; _, Z, U = state(winner['state_ref']); current = winner['loss_after']
        else: check(iteration == len(gradients), 'Continued after failed/zero direction')
    final, fz, fu = state(stage['final_state_ref'])
    check(used == len(trials) and accepted == steps and np.array_equal(fz, Z) and np.array_equal(fu, U), 'Final state is not last accepted state')
    check(path_length <= .5 + 128 * np.finfo(float).eps * max(1, Z.size), 'Four-update path-length budget exceeded')
    if not prep['no_information']:
        final_loss = verify_objective(stage['final_objective'], Z, U, prep, resolver, stage['final_objective_state_ref'])
        close(final_loss, current, 'Final objective selected a different training step')
        check(stage['final_objective']['forward_cache_reused'] is True and stage['final_objective']['backward_evaluation_count'] == 0, 'Final cache was charged as new work')
        reason = stage['stop_reason']; check(reason in ('MAX_ITERATIONS', 'ZERO_GRADIENT', 'TRIAL_BUDGET_EXHAUSTED', 'ZERO_FEASIBLE_DISPLACEMENT'), 'Unknown bounded stop reason')
        if reason == 'MAX_ITERATIONS': check(len(steps) == len(gradients) == 4, 'Premature max-iteration stop')
        if reason == 'ZERO_GRADIENT': check(gradients[-1]['gradient_norm'] == 0, 'False zero-gradient stop')
        if reason == 'TRIAL_BUDGET_EXHAUSTED': check(len(group) == 12 and not group[-1]['accepted'], 'False trial-budget exhaustion')
        if reason == 'ZERO_FEASIBLE_DISPLACEMENT':
            proposed = Z + .125 * .5 ** len(group) * direction; length = np.linalg.norm(proposed)
            projected = proposed if length <= .5 else proposed * (.5 / length)
            check(np.array_equal(projected, Z), 'False zero feasible displacement')
    prior = None
    if stage['mode'] == 'C_seq':
        bdata = resolver(b_stage['final_state_ref'])
        check(prep['final_problem']['prior_ref']==b_stage['final_state_ref']
              and stage['final_fit']['prior_ref']==b_stage['final_state_ref'],'Final C ref does not bind actual current B')
        for key, value in bdata.items(): check(np.array_equal(final['prior_B_' + key], value), 'Final C did not inherit exact actual B state: ' + key)
        check(np.array_equal(final['prior_old_class_indices'], [prep['classes'].index(value) for value in prep['old_classes']]), 'Actual B class-column map changed')
        prior = (bdata, prep['old_classes'])
    verify_head(stage['final_state_ref'], stage['final_fit'], resolver, prior)
    raw = {key: final[key] for key in BRANCHES}
    _array_close(H, _dictionary(raw), 'Prepared dictionary not bound to actual physical support')
    _array_close(final['V0'], dct_initial(), 'Fixed DCT dictionary changed')
    check(np.array_equal(final['singular_values'], coordinates['singular_values']), 'Final coordinate spectrum changed')
    if prior is not None:
        positions = [i for i, pid in enumerate(prep['training_physical_ids']) if pid in b_stage['training_physical_ids']]
        for key in BRANCHES: check(np.array_equal(final[key][positions], bdata[key]), 'Inherited old physical raw input changed: ' + key)
    close(stage['coordinate_norm'], float(np.linalg.norm(Z)), 'Final coordinate norm mismatch')
    check(stage['declared_coordinate_scalar_count'] == 736 * rank
          and stage['trainable_parameter_count'] == stage['optimizer_parameter_count'] == (0 if prep['no_information'] else 736 * rank)
          and stage['active_parameter_count'] == stage['trained_parameter_count'] == (736 * rank if steps else 0), 'Real active/trainable parameter accounting mismatch')
    analytic = final['alpha'].size + (final['intercept'] if stage['mode']=='B' else final['b']).size
    check(stage['analytic_head_parameter_count'] == analytic and stage['final_head_fit_count'] == 1
          and stage['final_factorization_count'] == stage['final_fit']['factorization_count'], 'Final analytic head/resource accounting mismatch')
    check(stage['analytic_coefficient_parameter_count']==len(prep['training_physical_ids'])*len(prep['classes'])
          and stage['analytic_intercept_parameter_count']==len(prep['classes'])
          and stage['changed_coordinate_scalar_count']==np.count_nonzero(Z), 'Analytic/adapter coordinates were conflated')
    check(type(stage['resident_numeric_state_bytes']) is int and type(stage['deployment_numeric_state_bytes']) is int
          and stage['resident_numeric_state_bytes']>=stage['deployment_numeric_state_bytes']>=len(prep['classes'])*8,
          'Resident/deployment numeric byte bounds invalid')
    for key in KERNEL_COUNTERS + ('raw_distance_evaluation_count','raw_distance_pair_count','reference_distance_evaluation_count',
        'reference_distance_pair_count','kernel_evaluation_count','kernel_pair_count','adapter_physical_evaluation_count') + tuple('head_' + s for s in SUFFIXES[1:]) + ('intercept_fit_count', 'intercept_addition_count', 'ajlr_forward_evaluation_count'):
        check(stage[key] == sum(obj[key] for obj in objectives) + stage['final_fit'][key], 'Full rejected-trial/final resource accounting mismatch: ' + key)
    for key in ('inner_objective_evaluation_count', 'inner_head_fit_count', 'inner_factorization_count', 'ce_adjoint_solve_count') + tuple('derivative_' + s for s in SUFFIXES[1:]):
        check(stage[key] == sum(obj[key] for obj in objectives), 'Full objective resource accounting mismatch: ' + key)
    for key in PEAK_COUNTERS:
        check(stage.get(key,0)==max([obj.get(key,0) for obj in objectives]+[stage['final_fit'].get(key,0)]),'Stage QP peak must be MAX: '+key)
    return final


def verify_score_workload(value, final, n):
    check(isinstance(value, dict) and value['score_physical_count'] == n and value['score_seconds'] >= 0, 'Missing actual outer inference workload')
    for component, data in (('residual', final), ('prior', {key[len('prior_B_'):]: item for key, item in final.items() if key.startswith('prior_B_')})):
        work = value[component]
        if not data:
            check(work == {}, 'B invented prior inference workload'); continue
        tau, gamma = _scalar(data, 'tau'), _scalar(data, 'gamma'); width, c = len(data['original_train_b']), data['alpha'].shape[1]
        calls = 0 if gamma is None else n * (1 + int(tau != 0 and bool(np.any(data['U']))))
        expected = dict(raw_distance_evaluation_count=calls, raw_distance_pair_count=calls * width,
            reference_distance_evaluation_count=calls, reference_distance_pair_count=calls * len(data['old_indices']),
            kernel_evaluation_count=n * int(gamma is not None), kernel_pair_count=n * width * int(gamma is not None),
            adapter_physical_evaluation_count=n * int(gamma is not None and tau != 0),
            dictionary_physical_evaluation_count=n * int(gamma is not None and tau != 0), intercept_addition_count=n * c)
        for key, expected_value in expected.items(): check(work.get(key, 0) == expected_value, 'Actual single-record score work mismatch: ' + component + '/' + key)
        check(work['score_seconds'] >= 0, 'Negative measured inference duration')
    for key in set(value['residual']) | set(value['prior']):
        if key != 'score_seconds': check(value[key] == value['residual'].get(key, 0) + value['prior'].get(key, 0), 'Outer prior/residual work total mismatch: ' + key)


def _verify_baseline(stage, entry, labels, registry, resolver):
    data = resolver(stage['final_state_ref']); ids = stage['training_physical_ids']; n, c = len(ids), len(registry)
    check(data['alpha'].shape == (n, c) and stage['optimizer_steps'] == 0
          and stage['all_states_estimated_from_trainfold_only'] is True and stage['source_validation'] is None, 'R0 physical/state boundary mismatch')
    y = np.asarray([registry.index(labels[pid]) for pid in ids]); d = _distances(data['original_train_b'], data['original_train_a'])
    s0 = float(np.sum(d[np.triu_indices(n, 1)]) / n)
    tau = None if c == 1 else float(np.median(np.min(np.where(y[:, None] != y[None, :], d, np.inf), axis=1)))
    close(_scalar(data, 'tau'), tau, 'R0 original bandwidth mismatch'); close(stage['interaction_centered_trace'], s0, 'R0 original trace mismatch')
    fact = int(c > 1 and s0 > 0); gamma = _scalar(data, 'gamma')
    if fact:
        rm1 = _radial(d, tau, 1.0, True); expected_gamma = s0 / float(-2 * np.sum(rm1[np.triu_indices(n, 1)]) / n)
        close(gamma, expected_gamma, 'R0 gamma mismatch'); K, _ = affine_analysis._center(rm1, rm1, np.full(n, 1 / n), gamma)
        # R0 has no free intercept.  Authenticate its actual reference-difference
        # deployment terms from its complete training geometry; no affine gauge
        # equivalence can justify changing the baseline centering measure.
        reference, reference_self = rm1[0], float(rm1[0, 0])
        diff = (rm1 - rm1[:, :1]) - reference[None, :] + reference_self
        mean = np.mean(diff, axis=0); grand = float(np.mean(mean))
        _array_close(data['reference_kernel'], reference, 'R0 reference kernel mismatch')
        close(_scalar(data, 'reference_self'), reference_self, 'R0 reference self mismatch')
        _array_close(data['center_mean'], mean, 'R0 complete-train center mean mismatch')
        close(_scalar(data, 'center_grand'), grand, 'R0 complete-train center grand mismatch')
        _array_close((K + np.eye(n)) @ data['alpha'], np.eye(c)[y] - 1 / c, 'Independent R0 ridge equation mismatch')
    else:
        check(gamma is None, 'Zero R0 invented scale'); _array_close(data['alpha'], np.zeros((n, c)), 'Zero R0 invented coefficients')
        _array_close(data['reference_kernel'], np.zeros(n), 'Zero R0 reference kernel mismatch')
        _array_close(data['center_mean'], np.zeros(n), 'Zero R0 center mean mismatch')
        check(_scalar(data, 'reference_self') == _scalar(data, 'center_grand') == 0., 'Zero R0 centering scalar mismatch')
    check(stage['factorization_calls'] == fact and stage['effective_degrees_of_freedom_extra_triangular_solves'] == 2 * fact,
          'R0 actual factor/EDF work mismatch')
    return fact


def _check_events(entry, by_prep, by_stage, binding):
    seen = {name: dict(initial=0, final=0, gradients=0, trials=0, steps=0) for name in by_stage}; prepared = set()
    for event in entry['training_events']:
        check(event['schema'] == SCHEMA and event['method'] == METHOD and event['objective_scope'] == 'INNER_SUPPORT_TRAINING_NOT_VALIDATION'
              and event['source_validation'] is None and event['scope'] == entry['scope'] and event['fold'] == entry['fold']
              and event['outer_trial'] == entry['trial'] and all(event.get(key) == value for key, value in binding.items()), 'Training event source/path binding mismatch')
        name, kind = event['state'], event['event']
        if kind == 'MARGIN_JOINT_PREPARED':
            pn = name.removesuffix('_prepare'); check(pn in by_prep and pn not in prepared, 'Duplicate/unknown preparation event')
            check(event['prepared_state_ref'] == by_prep[pn]['prepared_state_ref'] and event['prior_folds'] == by_prep[pn]['prior_folds'], 'Preparation event source/ref mismatch')
            for key in PREPARATION_COUNTERS:
                check(event[key]==by_prep[pn][key],'Preparation callback work mismatch: '+key)
            prepared.add(pn); continue
        check(name in by_stage, 'Unknown training stage'); stage = by_stage[name]; counts = seen[name]
        mapping = dict(MARGIN_JOINT_GRADIENT='gradients', MARGIN_JOINT_TRIAL='trials', MARGIN_JOINT_STEP='steps')
        if kind in mapping:
            key = mapping[kind]; index = counts[key]; check(index < len(stage[key]), 'Extra training event')
            check(all(event.get(field) == value for field, value in stage[key][index].items()), 'Full event/solver trace mismatch'); counts[key] += 1
        elif kind == 'MARGIN_JOINT_INITIAL':
            check(counts['initial'] == 0 and event['state_ref'] == stage['initialization_state_ref']
                  and event.get('objective') == stage.get('initial_objective'), 'Initial event/cache mismatch'); counts['initial'] += 1
        elif kind == 'MARGIN_JOINT_FINAL':
            check(counts['final'] == 0 and event['mode'] == stage['mode'] and event['state_ref'] == stage['final_state_ref'], 'FINAL real envelope mismatch')
            actual = event['audit']
            check(actual['final_state_ref'] == stage['final_state_ref'] and actual['preparation'] == stage['preparation']
                  and actual.get('final_objective') == stage.get('final_objective') and actual['stop_reason'] == stage['stop_reason'], 'Final audit/state mismatch')
            for key in STAGE_COUNTERS: check(actual[key] == stage[key], 'Final real work mismatch: ' + key)
            for key in PEAK_COUNTERS: check(actual.get(key,0)==stage.get(key,0),'Final callback peak MAX mismatch: '+key)
            counts['final'] += 1
        else: raise ValueError('Unknown Margin training event')
    check(prepared == set(by_prep) and all(v['initial'] == v['final'] == 1 and all(v[key] == len(by_stage[name][key])
          for key in ('gradients', 'trials', 'steps')) for name, v in seen.items()), 'Incomplete full solver event coverage')


def verify_record(record, split, old, resolver, expected_binding=None):
    finite_tree(record); binding = record['inheritance_binding']; labels = {pid: split['registered_classes'][int(y)] for pid, y in zip(split['support_ids'], split['support_labels'])}
    classes, old, k = sorted(split['registered_classes']), sorted(old), split['k']
    check(record['schema'] == SCHEMA and record['method'] == METHOD and record['scope'] == SCOPE
          and record['query_rows_used'] == record['source_rows_used'] == 0 and binding.get('split_id') == split['split_id'], 'Parent method/access/split mismatch')
    if expected_binding is not None: check(all(binding.get(key) == value for key, value in expected_binding.items()), 'Parent escaped current run/row')
    check(record['qp_resources']==dict(max_transitions=record['full_support']['preparations'][0]['qp_max_transitions'],
        max_factor_buffer_bytes=record['full_support']['preparations'][0]['qp_max_factor_buffer_bytes']) if k==1 else
        record['qp_resources']==dict(max_transitions=record['folds'][0]['preparations'][0]['qp_max_transitions'],
        max_factor_buffer_bytes=record['folds'][0]['preparations'][0]['qp_max_factor_buffer_bytes']),'Parent actual QP resources changed')
    check(all(record[key] == value for key, value in split_identity(split, old).items()), 'Parent declared identity mismatch')
    groups = {cls: sorted(pid for pid in labels if labels[pid] == cls) for cls in classes}
    check(len(labels) == len(split['support_ids']) and all(len(values) == k for values in groups.values())
          and record['classes'] == classes and record['old_classes'] == old and record['support_count'] == len(labels), 'Physical K/classes/IDs mismatch')
    expected = []
    if k == 1:
        check(record['fold_count'] == 0 and record['folds'] == [] and record['physical_fold_assignment'] == []
              and record['oof'] is record['oneshot_proxy'] is None and record['full_support'] is not None
              and record['heldout_unavailable_reason'] == 'K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT', 'True K1 independent held metric invented')
        expected.append((record['full_support'], set(labels), set(), 'support_full_k1', None))
    else:
        nfolds = min(k, 3); assignments = {pid: j % nfolds for values in groups.values() for j, pid in enumerate(values)}
        check(record['fold_count'] == len(record['folds']) == nfolds and record['full_support'] is None
              and {item['physical_id']: (item['class_id'], item['fold']) for item in record['physical_fold_assignment']}
              == {pid: (labels[pid], assignments[pid]) for pid in labels}, 'OOF physical assignment mismatch')
        for j, entry in enumerate(record['folds']):
            held = {pid for pid in labels if assignments[pid] == j}; expected.append((entry, set(labels) - held, held, 'support_oof', j))
        proxy = record['oneshot_proxy']; check(proxy['trial_count'] == len(proxy['trials']) == k and proxy['proxy_train_k'] == 1
            and proxy['aggregation'] == 'all_anchors_mean_within_parent_then_equal_parent', 'Complete proxy anchor coverage mismatch')
        for j, entry in enumerate(proxy['trials']):
            train = {values[j] for values in groups.values()}; expected.append((entry, train, set(labels) - train, 'support_oneshot_proxy', j))
    counts = dict.fromkeys(COUNTERS[4:], 0); logs, events, training = [], [], []; peak = 0
    for entry, train, held, scope, index in expected:
        coords = dict(binding, **{key: entry[key] for key in ('scope', 'fold', 'trial', 'parent_k', 'train_k')})
        check(entry['scope'] == scope and entry['parent_k'] == k and entry['train_k'] == len(train) // len(classes)
              and entry['held_k'] == len(held) // len(classes) and entry['fold'] == (index if scope == 'support_oof' else None)
              and entry['trial'] == (index if scope == 'support_oneshot_proxy' else None), 'Outer physical path coordinates mismatch')
        btrain, bheld = {pid for pid in train if labels[pid] in old}, {pid for pid in held if labels[pid] in old}
        for key, ids in (('b_training_ids', btrain), ('c_training_ids', train), ('b_ids', bheld), ('c_ids', held)):
            check(entry[key] == sorted(ids), 'Actual sorted physical path mismatch: ' + key)
        check(entry['held_labels'] == {pid: labels[pid] for pid in held} and entry['b_classes'] == old and entry['c_classes'] == classes
              and set(entry['paths']) == set(PATHS), 'Paired physical class/held mismatch')
        reuse = classes == old
        check(entry['c_reuses_b0'] is entry['c_reuses_b_candidates'] is reuse
              and [s['state'] for s in entry['stages']] == (['B0'] if reuse else ['B0', 'C0'])
              and [p['state'] for p in entry['preparations']] == (['B'] if reuse else ['B', 'C'])
              and [s['state'] for s in entry['candidate_stages']] == (['B_MARGIN'] if reuse else ['B_MARGIN', 'C_MARGIN_seq']), 'new0 exact-reuse stage structure mismatch')
        def refs(value):
            if isinstance(value, dict):
                if {'path', 'arrays', 'array_summaries'} <= set(value):
                    namespace = json.loads(value['namespace']); check(all(namespace.get(key) == item for key, item in coords.items()), 'Numeric archive escaped physical path')
                    check(namespace['state'] in ('OUTER_SUPPORT_HELD', 'B0', 'C0', 'B_prepare', 'C_prepare', 'B_MARGIN', 'C_MARGIN_seq'), 'Unknown archive state namespace')
                else:
                    for item in value.values(): refs(item)
            elif isinstance(value, list):
                for item in value: refs(item)
        refs(entry)
        for item in entry['stages'] + entry['preparations'] + entry['candidate_stages']:
            check(all(item[key] == value for key, value in coords.items()), 'Stage current-run/path binding mismatch')
        for item, keys in [(p, PREPARATION_COUNTERS) for p in entry['preparations']] + [(s, STAGE_COUNTERS) for s in entry['candidate_stages']]:
            check(all(type(item[key]) is int and item[key] >= 0 for key in keys), 'Invalid actual core counter type/value')
        raw = resolver(entry['outer_features_state_ref']) if held else None
        if held: check(set(raw) == set(BRANCHES) and all(len(value) == len(held) for value in raw.values())
                       and entry['prediction_status'] == 'FIXED_BEFORE_SUPPORT_TRUTH_JOIN', 'Fixed held feature/score evidence mismatch')
        else: check(entry['outer_features_state_ref'] is None and entry['prediction_status'] == 'NO_HELD_PREDICTIONS', 'True K1 held evidence invented')
        for stage in entry['stages']:
            is_b = stage['state'] == 'B0'; registry = old if is_b else classes; tids = entry['b_training_ids' if is_b else 'c_training_ids']
            check(stage['training_physical_ids'] == tids, 'R0 physical train binding mismatch')
            fact = _verify_baseline(stage, entry, labels, registry, resolver)
            counts['baseline_head_fit_count'] += 1; counts['baseline_factorization_count'] += fact
            counts['baseline_head_triangular_solve_count'] += 2 * fact; counts['baseline_effective_df_triangular_solve_count'] += 2 * fact
            logs.append(dict(event='BASE_FIT', **stage))
            if held:
                subset = [entry['c_ids'].index(pid) for pid in entry['b_ids']] if is_b else list(range(len(held)))
                scores = head_score(resolver(stage['final_state_ref']), {key: value[subset] for key, value in raw.items()})
                _array_close(entry['paths']['R0']['b_scores' if is_b else 'c_scores'], scores, 'Independent R0 outer fixed score mismatch')
                counts['final_score_evaluation_count'] += 1; counts['final_score_physical_count'] += len(subset)
        by_prep = {p['state']: p for p in entry['preparations']}; by_stage = {s['state']: s for s in entry['candidate_stages']}
        for prep, stage in zip(entry['preparations'], entry['candidate_stages']):
            check(stage['preparation_ref'] == prep['state'], 'Preparation/stage ordering mismatch')
            check(record['qp_resources']==dict(max_transitions=prep['qp_max_transitions'],max_factor_buffer_bytes=prep['qp_max_factor_buffer_bytes']),'Path changed declared QP resources')
            verify_preparation(prep, entry, labels, old, resolver); final = verify_candidate(stage, prep, entry, by_stage['B_MARGIN'], resolver)
            logs.extend((dict(event='MARGIN_PREPARATION', **prep), dict(event='CANDIDATE_FIT', **stage)))
            for key in PREPARATION_COUNTERS: counts[key] += prep[key]
            for key in STAGE_COUNTERS: counts[key] += stage[key]
            for key in PEAK_COUNTERS: counts[key] = max(counts[key],stage.get(key,0))
            counts['trained_margin_stage_count'] += int(stage['optimizer_steps'] > 0)
            if held:
                is_b = stage['mode'] == 'B'; subset = [entry['c_ids'].index(pid) for pid in entry['b_ids']] if is_b else list(range(len(held)))
                scores = head_score(final, {key: value[subset] for key, value in raw.items()})
                _array_close(entry['paths']['R_MARGIN_seq']['b_scores' if is_b else 'c_scores'], scores, 'Independent actual-B plus residual outer score mismatch')
                verify_score_workload(stage['score_workload'], final, len(subset))
                counts['final_score_evaluation_count'] += 1; counts['final_score_physical_count'] += len(subset)
            training.append(dict(split_id=record['split_id'], scope=scope, fold=entry['fold'], trial=entry['trial'], k=k,
                new_count=record['new_count'], state=stage['state'], train_k=entry['train_k'], latent_rank=prep['latent_rank'],
                trainable_parameter_count=stage['trainable_parameter_count'], trained_parameter_count=stage['trained_parameter_count'],
                optimizer_steps=stage['optimizer_steps'], stop_reason=stage['stop_reason'],
                initial_objective=stage.get('initial_objective'), final_objective=stage.get('final_objective'),
                gradients=stage['gradients'], trials=stage['trials'], steps=stage['steps'],
                initialization_state_ref=stage['initialization_state_ref'], final_state_ref=stage['final_state_ref'],
                final_objective_state_ref=stage.get('final_objective_state_ref'),final_fit=stage['final_fit'],
                qp_resources=dict(record['qp_resources']),actual_counters={key:stage[key] for key in STAGE_COUNTERS},
                QP_peaks={key:stage.get(key,0) for key in PEAK_COUNTERS},
                evidence_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION'))
        # Only now join the independently fixed scores to legal support-held truth.
        if held:
            evidence = {name: dict(**{key: entry[key] for key in ('b_ids', 'b_classes', 'c_ids', 'c_classes', 'held_labels')},
                b_scores=entry['paths'][name]['b_scores'], c_scores=entry['paths'][name]['c_scores']) for name in PATHS}
            fresh = assess_paths(evidence, old)
            for name in PATHS: check(entry['paths'][name] == dict(b_scores=evidence[name]['b_scores'], c_scores=evidence[name]['c_scores'], **fresh[name]), 'Fixed-score post-truth diagnostics mismatch')
        else:
            for path in entry['paths'].values(): check(path['b_scores'] == path['c_scores'] == [] and path['diagnostic'] is path['held_comparisons'] is None
                and path['metrics'] == dict.fromkeys(METRICS), 'True K1 metrics fabricated')
        _check_events(entry, by_prep, by_stage, binding); events.extend(entry['training_events']); counts['sequence_paths'] += 1
        if reuse:
            for name in PATHS: check(entry['paths'][name]['b_scores'] == entry['paths'][name]['c_scores'], 'new0 changed actual B scores')
        final_stage = by_stage['B_MARGIN' if reuse else 'C_MARGIN_seq']
        check(entry['deployment_C_state_bytes'] == {'R_MARGIN_seq': final_stage['resident_numeric_state_bytes']}
              and entry['minimum_deployment_numeric_state_bytes'] == {'R_MARGIN_seq': final_stage['deployment_numeric_state_bytes']}, 'Resident/minimum deployment byte scope mismatch')
        peak = max(peak, final_stage['resident_numeric_state_bytes'])
    counts['candidate_preparation_count'] = counts['ajlr_preparation_count']; counts['candidate_stage_count'] = counts['ajlr_stage_count']
    counts['baseline_triangular_solve_count'] = counts['baseline_head_triangular_solve_count'] + counts['baseline_effective_df_triangular_solve_count']
    counts['head_fit_count'] = sum(counts[key] for key in ('baseline_head_fit_count', 'inner_head_fit_count', 'final_head_fit_count', 'prior_head_fit_count'))
    counts['factorization_count'] = sum(counts[key] for key in ('baseline_factorization_count', 'inner_factorization_count', 'final_factorization_count', 'prior_factorization_count'))
    check(all(type(record[key]) is int and record[key] == value for key, value in counts.items()) and record['persistent_state_bytes'] == peak, 'Parent actual work aggregate mismatch')
    if k > 1:
        check(record['oof'] == dict(paths=pooled_assess(record['folds'], labels, classes, old), aggregation='one_record_per_physical_held_id'), 'OOF physical pooling mismatch')
        check(record['oneshot_proxy']['parent_mean_metrics'] == parent_mean(record['oneshot_proxy']['trials']), 'Proxy parent-first H/gap/decline aggregation mismatch')
    resolver.verify_tree(record)
    return logs, events, dict(oof=None if k == 1 else {name: record['oof']['paths'][name]['metrics'] for name in PATHS},
        proxy=None if k == 1 else record['oneshot_proxy']['parent_mean_metrics']), training


def completion_check(done, spec):
    """Require exactly the declared dynamic rows/physical matrix before traces."""
    count = len(spec['rows']); exact = budget_for_spec(spec)['total']['exact']
    check(done['status'] == STATUS and done['model_rows'] == done['completed_rows'] == count
          and done['run_id'] == spec['run_id'], 'Declared row matrix incomplete')
    for key, value in exact.items(): check(done[key] == value, 'Incomplete declared physical coverage: ' + key)


def verify_stage_stream(stream, logs, split_id):
    for value in logs:
        check(next(stream, None) == dict(compact_event(value), schema=SCHEMA, method=METHOD, split_id=split_id), 'Actual fit stage stream mismatch')


def accumulate_resources(resources, record, logs):
    resources['parent_wall_seconds_sum'] = resources.get('parent_wall_seconds_sum', 0.) + record['fit_seconds']
    for row in logs:
        group = row['event'].lower()
        for key, value in row.items():
            if key.endswith('_seconds') and value is not None:
                check(value >= 0, 'Negative measured duration'); name = group + '_' + key + '_sum'; resources[name] = resources.get(name, 0.) + value
            if key.endswith('_bytes') and type(value) is int:
                name = group + '_maximum_' + key; resources[name] = max(resources.get(name, 0), value)
        if row['event'] == 'CANDIDATE_FIT':
            forwards = ([] if row.get('initial_objective') is None else [row['initial_objective']]) + [v['objective'] for v in row['trials']]
            backwards = [v['objective'] for v in row['gradients']]
            for key, values in (('objective_forward_seconds_sum', forwards), ('objective_backward_seconds_sum', backwards)):
                resources[key] = resources.get(key, 0.) + sum(value['objective_seconds'] for value in values)
            resources['final_head_forward_seconds_sum'] = resources.get('final_head_forward_seconds_sum', 0.) + row['final_fit']['forward_seconds']
            for component in ('residual', 'prior'):
                for key, value in (row.get('score_workload') or {}).get(component, {}).items():
                    if type(value) in (int, float):
                        name = 'outer_' + component + '_' + key + '_sum'; resources[name] = resources.get(name, 0) + value


def _add(groups, key, metrics):
    group = groups.setdefault(key, {metric: [] for metric in METRICS})
    for metric, value in metrics.items(): group[metric].append(value)


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows(csv_record({key: row.get(key) for key in fields}) for row in rows)


def _resolved_argv(spec, row):
    co = spec['probe']['cohorts'][row['cohort']]
    return [str(PurePosixPath(spec['code']['cwd']) / 'tools/evaluate_d92_margin_joint_probe.py'),
        '--support-features', row['support_features'], '--capsule', co['capsule'], '--output', row['output_root'] + '/probe',
        '--config', co['evaluation_config'], '--expected-capsule-id', co['capsule_id'],
        '--expected-checkpoint-sha256', row['expected_checkpoint_sha256'], '--expected-model-seed', str(row['seeds']['model']),
        '--run-id', spec['run_id'], '--row-id', row['row_id']]


def summarize(*, spec, run_root=None, output):
    analysis_started=time.perf_counter()
    spec = read(spec) if not isinstance(spec, dict) else spec; validate_spec(spec)
    root, out = Path(run_root or spec['execution']['remote_run_root']), Path(output)
    if out.exists(): raise FileExistsError(out)
    launch, done, state = [read(root / name) for name in ('startup.json', 'complete.json', 'state.json')]
    completion_check(done, spec)
    check(launch['spec'] == spec and launch['commit'] == done['commit'] and set(state) == {row['row_id'] for row in spec['rows']}
          and all(value['status'] == STATUS for value in state.values()), 'Complete run/source commit/row state mismatch')
    check(all(value['query_access'] is False and value['source_sample_access'] is False for value in (launch, done)), 'Forbidden run access')
    expected_supervisor = [str(PurePosixPath(spec['code']['cwd']) / 'tools/run_d92_margin_joint_probe.py'),
        '--spec', str(PurePosixPath(spec['code']['cwd']) / spec['spec_path']), '--commit', done['commit']]
    check(launch['argv'] == expected_supervisor, 'Supervisor resolved argv mismatch')
    lanes = []
    # Complete bindings/inventories precede every fixed-score/held trace read.
    for row in spec['rows']:
        co = spec['probe']['cohorts'][row['cohort']]; lane = root / row['row_id'] / 'probe'
        marker = verify_marker(lane / 'probe_complete.json', spec, row); startup = read(lane / 'startup.json'); check_bind(startup, row, co)
        expected = _resolved_argv(spec, row); process = read(root / row['row_id'] / 'probe.log.process.json')
        check(startup['argv'] == expected and startup['python'] == spec['code']['environment']
              and process['argv'] == [startup['python'], '-u', *expected] and process['pid'] == startup['pid']
              and process['cwd'] == spec['code']['cwd'] and process['kind'] == 'probe', 'Evaluator process/resolved argv/core path mismatch')
        check(startup['run_id'] == marker['run_id'] == spec['run_id'] and startup['row_id'] == marker['row_id'] == row['row_id']
              and startup['schema'] == marker['schema'] == SCHEMA and startup['method'] == marker['method'] == METHOD
              and startup['scope'] == marker['scope'] == SCOPE, 'Row schema/actual run binding mismatch')
        expected_config = dict(algorithm=PROBE_CONFIG, producer_matrix=co['matrix'], selection=co['selection'],qp_resources=spec['probe']['qp_resources'])
        check(startup['config'] == expected_config and startup['episodes'] == spec['probe']['budget']['per_row'][row['row_id']]['exact']['episodes']
              and startup['producer_episodes'] == co['expected_split_count'], 'Dynamic selected axes/config coverage mismatch')
        for value in (startup, marker):
            check(all(value['channel'][key] == expected_value for key, expected_value in CHANNEL.items()) and value['scenarios'] == SCENARIOS
                  and value['query_rows_used'] == value['source_rows_used'] == 0 and value['truth_read'] is False, 'Channel/access mismatch')
            check(value['payload_audit']['new_source_payload_bytes'] == value['payload_audit']['new_ground_statistics_bytes'] == 0, 'Additional forbidden source payload')
        check(all(startup[key] is False for key in ('query_iq_access', 'checkpoint_loaded', 'encoder_updated'))
              and startup['actual_A'] is None and startup['objective_scope'] == 'INNER_SUPPORT_TRAINING_NOT_VALIDATION', 'Invented A/forbidden fit access')
        cache_root = Path(startup['support_features']); check(cache_root == Path(row['support_features']), 'Declared cache reference changed')
        feature, plan, provenance = [read(cache_root / name) for name in ('features_complete.json', 'support_splits.json', 'checkpoint_provenance.json')]
        check_bind(feature, row, co)
        for value in (feature, plan): check(value['capsule_id'] == co['capsule_id'] and value['checkpoint_sha256'] == row['expected_checkpoint_sha256'], 'Producer cache/capsule binding mismatch')
        check(feature['status'] == 'BRANCH_SUPPORT_FEATURES_COMPLETE' and feature['split_count'] == co['expected_split_count']
              and provenance == startup['provenance'] and provenance['verdict'] == 'MATCHED_SOURCE_ONLY_SCRATCH'
              and provenance['target_access_before_freeze'] is False and provenance['checkpoint_inheritance'] == [], 'Source-only contract/provenance mismatch')
        old = feature['classes']; check(len(old) == 6, 'Actual six-old registry required')
        chosen = selected_tasks([(split, None, None) for split in plan['splits']], co['selection'], old)
        verify_artifact_inventory(lane, marker); resolver = StateResolver(lane)
        lanes.append((row, lane, marker, {split['split_id']: split for split, _, _ in chosen}, old, startup, resolver))
    coverage = dict.fromkeys(COUNTERS, 0); resources = {}; training = []; archives = []
    strata = {name: {} for name in ('overall', 'by_k_new_count', 'by_receiver_scene', 'by_model_cohort')}
    resource_strata = {name: {} for name in ('by_k_new_count', 'by_receiver_scene', 'by_model_cohort', 'by_row')}
    old_by_k = {}
    for row, lane, marker, expected, old, startup, resolver in lanes:
        seen = set(); counts = dict.fromkeys(COUNTERS, 0); stages = iter(jsonlines(lane / 'fit_stages.jsonl'))
        full_events = iter(jsonlines(lane / 'training_events.jsonl')); small_events = iter(jsonlines(lane / 'training_events_compact.jsonl'))
        for record, small in itertools.zip_longest(jsonlines(lane / 'fit_trace.jsonl'), jsonlines(lane / 'compact.jsonl')):
            check(record is not None and small is not None and record['split_id'] in expected and record['split_id'] not in seen, 'Duplicate/unexpected/missing parent trace')
            sid = record['split_id']; seen.add(sid); split = expected[sid]
            labels = dict(zip(split['support_ids'], split['support_labels']))
            old_ids = tuple(sorted(pid for pid, y in labels.items() if split['registered_classes'][int(y)] in old))
            old_key = (row['row_id'], split['receiver'], split['scenario'], split['k'], split['support_seed'])
            check(old_by_k.setdefault(old_key, old_ids) == old_ids, 'Old physical support changed across new-count rows')
            logs, events, measurements, trace = verify_record(record, split, old, resolver, dict(run_id=spec['run_id'], row_id=row['row_id'], split_id=sid))
            check(small == compact_record(record), 'Compact parent evidence mismatch'); verify_stage_stream(stages, logs, sid)
            for event in events:
                value = dict(event, split_id=sid)
                check(next(full_events, None) == value and next(small_events, None) == compact_event(value), 'Full/compact training event inventory mismatch')
            training.extend(dict(row_id=row['row_id'], model_seed=row['seeds']['model'], cohort=row['cohort'], receiver=record['receiver'], scenario=record['scenario'], **value) for value in trace)
            accumulate_resources(resources, record, logs)
            keys = dict(by_k_new_count=(record['k'], record['new_count']), by_receiver_scene=(row['cohort'], record['receiver'], record['scenario'], record['k'], record['new_count']),
                        by_model_cohort=(row['seeds']['model'], row['cohort'], record['k'], record['new_count']), by_row=(row['row_id'],))
            for name, key in keys.items():
                cell = resource_strata[name].setdefault(key, {}); accumulate_resources(cell, record, logs); cell['parents'] = cell.get('parents', 0) + 1
                for counter in COUNTERS[4:]: cell[counter] = max(cell.get(counter,0),record[counter]) if counter in PEAK_COUNTERS else cell.get(counter,0)+record[counter]
            counts['episodes'] += 1; counts['k1_episodes'] += int(record['k'] == 1); counts['oof_episodes'] += int(record['k'] > 1)
            counts['proxy_anchor_count'] += small['proxy_anchor_count']
            for counter in COUNTERS[4:]: counts[counter] = max(counts[counter],record[counter]) if counter in PEAK_COUNTERS else counts[counter]+record[counter]
            for diagnostic, paths in measurements.items():
                if paths is None: paths = {path: dict.fromkeys(METRICS) for path in PATHS}
                population = 'old_only' if record['new_count'] == 0 else 'new_present'
                for path, values in paths.items():
                    _add(strata['overall'], (diagnostic, path, population), values)
                    _add(strata['by_k_new_count'], (diagnostic, path, record['k'], record['new_count']), values)
                    _add(strata['by_receiver_scene'], (diagnostic, path, row['cohort'], record['receiver'], record['scenario'], record['k'], record['new_count']), values)
                    _add(strata['by_model_cohort'], (diagnostic, path, row['seeds']['model'], row['cohort'], record['k'], record['new_count']), values)
        check(seen == set(expected) and next(stages, None) is next(full_events, None) is next(small_events, None) is None, 'Missing declared parents or extra events')
        resolver.finalize(); manifest = resolver.manifest
        check(marker['state_archive_file_count'] == manifest['file_count'] and marker['state_archive_file_bytes'] == manifest['total_file_bytes']
              and marker['state_archive_numeric_bytes'] == manifest['numeric_array_bytes'], 'Marker actual archive byte mismatch')
        check(all(counts[key] == marker[key] == state[row['row_id']][key] for key in COUNTERS), 'Row actual count mismatch')
        for key in COUNTERS: coverage[key] = max(coverage[key],counts[key]) if key in PEAK_COUNTERS else coverage[key]+counts[key]
        resolver.clear_cache(); archives.append(dict(row_id=row['row_id'], root=str(lane), manifest=str(lane / 'state_manifest.json'),
            file_count=manifest['file_count'], file_bytes=manifest['total_file_bytes'], numeric_array_bytes=manifest['numeric_array_bytes'],
            archive_seconds=manifest['archive_seconds'], by_phase=manifest['by_phase'], cache=resolver.cache_statistics(), independent_analysis_work=dict(resolver.analysis_work)))
    check(all(coverage[key] == done[key] for key in COUNTERS), 'Run actual count mismatch')
    budget = budget_for_spec(spec)['total']
    check(all(coverage[key] == value for key, value in budget['exact'].items()) and all(coverage[key] <= value for key, value in budget['maximum'].items()), 'Dynamic structural budget exceeded')
    check(coverage['candidate_stage_count'] == len(training) and done['finished'] >= launch['started'], 'Complete training-stage/wall evidence mismatch')
    resources.update(actual_counters=dict(coverage), structural_budget=budget, run_wall_seconds=done['finished'] - launch['started'],
        lane_wall_seconds_sum=sum(marker['wall_seconds'] for _, _, marker, _, _, _, _ in lanes),
        maximum_lane_peak_rss_bytes=max((marker['peak_process_rss_bytes'] for _, _, marker, _, _, _, _ in lanes if marker['peak_process_rss_bytes'] is not None), default=None),
        peak_gpu_memory_bytes=None, deployment_package_bytes=None, incremental_transmission_bytes=None,
        additional_ground_data_payload_bytes=0, additional_ground_statistics_bytes=0, separate_spectral_seconds=None,
        independent_analysis_work={key: sum(item['independent_analysis_work'][key] for item in archives) for key in archives[0]['independent_analysis_work']},
        triangular_rhs_count_scope='Per actual system: columns=r; elements=n*r; dense=n*n*r; proxy not measured FLOPs',
        resident_state_scope='Core resident numeric buffers include actual B; archive bytes, minimum deployment and process RSS are distinct. Alias layouts cannot be reconstructed from NPZ copies.',
        positive_kernel_evidence='Independently rebuilt full physical raw Gaussian/equivalence Gram, PSD spectra, primal/dual/KKT/all physical inequalities and saved affine/active factor residuals',
        unmeasured_reason='CPU only; no deployment transport/energy measured; candidate raw-kernel spectra are charged in actual head timing, not separately timed',
        hardware=[dict(row_id=row['row_id'], hardware=startup['hardware'], blas_environment=startup['blas_environment']) for row, _, _, _, _, startup, _ in lanes])
    dimensions = dict(overall=('diagnostic', 'path', 'population'), by_k_new_count=('diagnostic', 'path', 'k', 'new_count'),
        by_receiver_scene=('diagnostic', 'path', 'cohort', 'receiver', 'scenario', 'k', 'new_count'), by_model_cohort=('diagnostic', 'path', 'model_seed', 'cohort', 'k', 'new_count'))
    tables = {name: _statistics(groups, dimensions[name]) for name, groups in strata.items()}
    rdimensions = dict(by_k_new_count=('k', 'new_count'), by_receiver_scene=('cohort', 'receiver', 'scenario', 'k', 'new_count'),
                       by_model_cohort=('model_seed', 'cohort', 'k', 'new_count'), by_row=('row_id',))
    resource_tables = {name: [dict(zip(rdimensions[name], key), **value) for key, value in sorted(groups.items())] for name, groups in resource_strata.items()}
    summary = dict(status=SUMMARY_STATUS, summary_schema=SUMMARY_SCHEMA, schema=SCHEMA, method=METHOD, scope=SCOPE,
        run_id=spec['run_id'], release_commit=done['commit'], model_rows=len(spec['rows']), coverage=coverage, resources=resources,
        resource_statistics=resource_tables, statistics=tables, algorithm=PROBE_CONFIG, qp_resources=spec['probe']['qp_resources'],channel=CHANNEL, old_class_count=6,
        actual_A=None, adaptation_gain_B_minus_A=None, automatic_promotion=False, performance_gate=None,
        analysis_wall_seconds=time.perf_counter()-analysis_started,
        analysis_time_scope='SPEC_AND_COMPLETE_BINDING_ARCHIVE_AND_ALL_MATH_VERIFICATION_BEFORE_OUTPUT_WRITES',
        query_rows_used=0, source_rows_used=0, training_stage_count=len(training), state_archives=archives,
        state_archive_file_count=sum(value['file_count'] for value in archives), state_archive_file_bytes=sum(value['file_bytes'] for value in archives),
        raw_training_sources=[dict(row_id=row['row_id'], fit_trace=str(lane / 'fit_trace.jsonl'), full_training_events=str(lane / 'training_events.jsonl'),
            compact_training_events=str(lane / 'training_events_compact.jsonl')) for row, lane, _, _, _, _, _ in lanes],
        interpretation=['Support-only OOF and proxy; no query accessed. Legal held truth is joined after independently fixed scores.',
            'B/C optimize pooled class RMS CE only; no proximal +Z gradient. Ball .5 and 4x12 trials use actual projected delta; final is the last accepted state.',
            'All old physical support and every registered output column are constrained. Actual B mapping, fold, classes and physical IDs remain frozen; new0 reuses B.',
            'True K1 fits full closed heads and has N/A independent held metrics. Proxy is a separate within-parent diagnostic.',
            'H, gap and decline are computed within each parent before parent averaging. Proxy anchors are averaged within parent first. Repeated old support across new counts is not independent replication.',
            'A and B-A remain N/A until a legal actual ground packet is bound; R0/B0 is not A.',
            'Every actual QP factor attempt, transition, scan and RHS is charged; spectra occur once per C head, never a fixed Conditional formula. Rejected trials and priors are included.',
            'Independent general saddle/active solves and raw-kernel spectra add analysis work separately from the original training ledger.',
            'No accuracy threshold, promotion, training-peak selection or performance feedback changes method state.'])
    out.mkdir(parents=True, exist_ok=False); write_json(out / 'summary.json', summary)
    for name, rows in tables.items(): _write_csv(out / (name + '.csv'), rows)
    for name, rows in resource_tables.items(): _write_csv(out / ('resources_' + name + '.csv'), rows)
    with (out / 'training_objectives.jsonl').open('x', encoding='utf-8') as stream:
        for row in training: stream.write(json.dumps(row, allow_nan=False) + '\n')
    _write_csv(out / 'training_objectives.csv', [compact_event(row) for row in training])
    cells = {}
    for row in tables['by_k_new_count']: cells.setdefault(tuple(row[key] for key in dimensions['by_k_new_count']), {})[row['metric']] = row['mean']
    fields = ('B_old_accuracy', 'C_old_accuracy', 'C_new_accuracy', 'C_h', 'total_old_accuracy_drop', 'C_abs_new_old_gap')
    lines = ['# MarginJoint 支持集独立分析', '', 'A 与 B−A 为 N/A。准确率按百分数、差值按百分点。真实 K1 的独立 held 为 N/A；proxy 单列。', '',
        '| 诊断 | 路径 | K | 新类数 | B 旧 | C 旧 | C 新 | H | 注册下降 | 新旧差 |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key, values in sorted(cells.items()): lines.append('| ' + ' | '.join([str(value) for value in key] + ['N/A' if values[field] is None else f'{100 * values[field]:.3f}' for field in fields]) + ' |')
    lines += ['', *['- ' + value for value in summary['interpretation']], '']; (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--run-root'); parser.add_argument('--output', required=True)
    result = summarize(**vars(parser.parse_args())); print(json.dumps(dict(status=result['status'], coverage=result['coverage'])))


if __name__ == '__main__': main()
