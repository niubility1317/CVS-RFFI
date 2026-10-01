"""Literal dictionary certificates only; no stored model dictionary is read."""
import ast
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__' and sys.argv[1:]==['--static-only']:
    for path in (ROOT/'code/cvsrffi/d92_support_metric_basis.py',Path(__file__)):
        source=path.read_bytes().decode('utf-8',errors='strict');ast.parse(source,filename=str(path))
        assert not any(ord(c)<32 and c not in '\n\r\t' for c in source)
    print('UTF8/AST PASS: 2 owned basis files; no numerical imports/execution')
    raise SystemExit(0)

from decimal import Decimal,localcontext
from fractions import Fraction
import json

sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import pytest
from cvsrffi.d92_support_metric_basis import build_support_metric_basis,SupportMetricBasisFailure,ENCLOSURE_BITS

# Literal technical budgets for these small fixtures, not experiment defaults.
LIMITS=dict(max_integer_bits=32768,max_fraction_operations=200000)


def build(Q,**limits):return build_support_metric_basis(Q=np.asarray(Q,dtype=np.float64),**dict(LIMITS,**limits))


def dot(a,b):return sum((x*y for x,y in zip(a,b)),Fraction(0))


def check_exact_state(Q,state):
    columns=[tuple(Fraction.from_float(float(x)) for x in Q[:,j]) for j in range(Q.shape[1])]
    for j,(v,rep) in enumerate(zip(state.exact_basis,state.exact_residual_coefficients)):
        assert all(v[i]==sum((columns[k][i]*rep[k] for k in range(len(columns))),Fraction(0)) for i in range(len(Q)))
        assert dot(v,v)==state.exact_norm_squared[j]>0
        assert all(dot(v,old)==0 for old in state.exact_basis[:j])
    for original,relation in state.exact_zero_relations:
        assert relation[original]==1
        assert all(sum((columns[k][i]*relation[k] for k in range(len(columns))),Fraction(0))==0 for i in range(len(Q)))


def check_enclosures(Q,state):
    # Independent high-precision sqrt oracle; the certificate itself uses isqrt.
    with localcontext() as ctx:
        ctx.prec=250
        squared_error=Decimal(0)
        for j,v in enumerate(state.exact_basis):
            norm=state.exact_norm_squared[j]
            root=(Decimal(norm.numerator)/Decimal(norm.denominator)).sqrt()
            for i,x in enumerate(v):
                true=(Decimal(x.numerator)/Decimal(x.denominator))/root
                lower=Decimal.from_float(float(state.basis_lower[i,j]));upper=Decimal.from_float(float(state.basis_upper[i,j]))
                # Decimal's oracle is independently rounded; allow its own final
                # 250-digit resolution only in checking exact endpoint cases.
                oracle_error=Decimal('1e-245')*max(Decimal(1),abs(true))
                assert lower-oracle_error<=true<=upper+oracle_error
                difference=abs(Decimal.from_float(float(state.basis[i,j]))-true)
                assert difference<=Decimal.from_float(float(state.error_bounds[i,j]))+oracle_error
                squared_error+=difference*difference
                exact_lower=state.exact_unit_lower[j][i];exact_upper=state.exact_unit_upper[j][i]
                assert Fraction.from_float(float(state.basis_lower[i,j]))<=exact_lower<=exact_upper<=Fraction.from_float(float(state.basis_upper[i,j]))
        assert squared_error.sqrt()<=Decimal.from_float(state.spectral_error_bound)+Decimal('1e-240')
    measured=np.linalg.norm(state.basis.T@state.basis-np.eye(state.rank),ord=2) if state.rank else 0.
    # This measured Gram itself has float rounding, separate from certified U error.
    assert measured<=state.orthogonality_error_bound+32*np.finfo(float).eps*max(1,Q.shape[0],state.rank)


@pytest.mark.parametrize('shape',[(0,0),(0,5),(3,0),(160,5)])
def test_zero_and_empty_exact_rank(shape):
    Q=np.zeros(shape,dtype=np.float64);s=build(Q)
    assert s.rank==0 and s.column_indices==() and s.basis.shape==(shape[0],0)
    assert len(s.exact_zero_relations)==shape[1]
    assert s.spectral_error_bound==s.orthogonality_error_bound==0
    check_exact_state(Q,s)


def test_fixed_order_exact_dependence_and_certificate():
    Q=np.array([[1,2,0,1,0],[2,4,1,3,0],[3,6,-1,2,0],[4,8,2,6,0]],dtype=np.float64)
    s=build(Q)
    assert s.rank==2 and s.column_indices==(0,2)
    assert [i for i,_ in s.exact_zero_relations]==[1,3,4]
    check_exact_state(Q,s);check_enclosures(Q,s)
    audit=s.audit_dict();certificate=json.loads(s.certificate_json())
    assert certificate['rank']==2 and certificate['input_scope']=='EXACT_STORED_BINARY64_NOT_PRE_ROUNDING_PROTOTYPES'
    assert audit['certificate_utf8_bytes']==len(s.certificate_json().encode('utf-8'))
    assert audit['numeric_array_payload_bytes']==sum(a.nbytes for a in (s.Q,s.basis,s.basis_lower,s.basis_upper,s.error_bounds))
    assert audit['logical_rational_integer_payload_bytes']>0 and audit['process_peak_memory_bytes'] is None
    assert audit['fraction_operations_completed']==sum(audit['fraction_operations_by_kind'].values())
    assert audit['integer_operations_completed']==sum(audit['integer_operations_by_kind'].values())
    assert audit['fraction_operations_attempted']==audit['fraction_operations_completed']
    assert audit['columns_started']==audit['columns_completed']==5 and audit['enclosure_bits']==128
    assert audit['integer_operations_by_kind']['isqrt']==s.rank+1


def test_smallest_nonzero_input_is_retained_before_float_norm_underflow():
    tiny=np.nextafter(np.float64(0),np.float64(1))
    Q=np.array([[1,1],[0,tiny],[0,0]],dtype=np.float64)
    s=build(Q)
    assert s.rank==2 and s.column_indices==(0,1)
    assert s.exact_norm_squared[1]==Fraction.from_float(float(tiny))**2
    np.testing.assert_array_equal(s.basis,np.array([[1.,0.],[0.,1.],[0.,0.]]))
    check_exact_state(Q,s);check_enclosures(Q,s)


def test_cancellation_residual_smaller_than_binary64_subnormal_is_kept():
    tiny=np.nextafter(np.float64(0),np.float64(1))
    # Projecting (tiny,0) on (1,1) leaves exact (+tiny/2,-tiny/2).
    Q=np.array([[1,tiny],[1,0]],dtype=np.float64);s=build(Q)
    assert s.rank==2
    assert s.exact_basis[1]==(Fraction.from_float(float(tiny))/2,-Fraction.from_float(float(tiny))/2)
    assert all(float(v)==0. for v in s.exact_basis[1])
    assert np.all(np.abs(s.basis[:,1])>.7)
    check_exact_state(Q,s);check_enclosures(Q,s)


def test_negative_components_irrational_norm_and_subnormal_enclosure():
    Q=np.array([[1.,1.],[-2.,3.],[5.,-4.],[np.nextafter(0.,1.),0.]],dtype=np.float64)
    s=build(Q);check_exact_state(Q,s);check_enclosures(Q,s)
    assert np.any(s.basis_lower<s.basis_upper)
    assert s.audit_dict()['nextafter_steps']>0


def test_square_root_rational_enclosures_exact_integer_readback():
    state=build([[1.,0.],[2.,1.],[3.,-1.]])
    certificate=json.loads(state.certificate_json())
    def decode(v):return Fraction(int(v['numerator_hex'],16),int(v['denominator_hex'],16))
    for value in certificate['normalizations']:
        S=decode(value['scaled_norm_squared']);lo=decode(value['sqrt_lower']);hi=decode(value['sqrt_upper'])
        assert lo*lo<=S<=hi*hi and hi-lo<=Fraction(1,1<<ENCLOSURE_BITS)
        assert S>=1


def test_inputs_outputs_and_exact_state_do_not_alias_or_rebind():
    Q=np.eye(3,dtype=np.float64);before=Q.copy();s=build(Q)
    np.testing.assert_array_equal(Q,before);Q[0,0]=8
    assert s.Q[0,0]==1
    for a in (s.Q,s.basis,s.basis_lower,s.basis_upper,s.error_bounds):
        with pytest.raises(ValueError):a.flags.writeable=True
    with pytest.raises(AttributeError):s.rank=0
    with pytest.raises(TypeError):s.exact_basis[0][0]=Fraction(0)
    audit=s.audit;audit['fraction_operations_by_kind']['add']=-1
    assert s.audit_dict()['fraction_operations_by_kind']['add']>=0


@pytest.mark.parametrize('Q',[np.array([[np.nan]]),np.array([[np.inf]]),np.zeros((161,1)),np.zeros((1,6)),np.zeros(4),np.zeros((1,1),dtype=np.float32),np.ones((1,1),dtype=np.int64)])
def test_invalid_inputs_rejected_before_computation(Q):
    with pytest.raises(ValueError):build_support_metric_basis(Q=Q,**LIMITS)


@pytest.mark.parametrize('key,value',[('max_integer_bits',True),('max_fraction_operations',0),('max_integer_bits',1.5)])
def test_explicit_positive_resource_guards(key,value):
    with pytest.raises(ValueError):build_support_metric_basis(Q=np.eye(2),**dict(LIMITS,**{key:value}))


def test_operation_exhaustion_keeps_partial_exact_columns_not_false_rank():
    with pytest.raises(SupportMetricBasisFailure) as err:build(np.eye(3),max_fraction_operations=15)
    exc=err.value;a=exc.audit_dict()
    assert exc.code=='FRACTION_OPERATION_LIMIT' and a['status']=='TECHNICAL_FAILURE' and a['exact_rank_complete'] is False
    assert a['fraction_operations_attempted']==16 and a['fraction_operations_completed']==15
    assert a['completed_independent_columns']>=1
    assert exc.exact_state['column_indices'] and np.array_equal(exc.arrays['Q'],np.eye(3))


def test_integer_limit_is_explicit_technical_failure_and_no_truncation():
    Q=np.array([[1.,1.],[0.,np.nextafter(0.,1.)]])
    with pytest.raises(SupportMetricBasisFailure) as err:build(Q,max_integer_bits=512)
    assert err.value.code=='INTEGER_BIT_LIMIT'
    assert err.value.audit_dict()['max_observed_integer_bits']>512
    assert err.value.audit_dict()['exact_rank_complete'] is False


def test_deterministic_certificate_and_orthogonal_rank_without_svd(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No numerical rank or pseudoinverse')
    monkeypatch.setattr(np.linalg,'svd',forbidden);monkeypatch.setattr(np.linalg,'pinv',forbidden)
    Q=np.array([[1.,4.,5.],[2.,1.,3.],[3.,0.,3.]])
    a=build(Q);b=build(Q)
    assert a.certificate_json()==b.certificate_json()
    assert a.audit_dict()['fraction_operations_by_kind']==b.audit_dict()['fraction_operations_by_kind']
    assert a.column_indices==(0,1) and a.rank==2
