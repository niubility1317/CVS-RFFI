"""Exact rank of stored binary64 dictionaries and enclosed physical bases.

No model, file I/O, head, pseudoinverse or numerical rank threshold is used.
Integer limits bound conservative arithmetic operand/intermediate bit estimates;
binary64 decoding itself has the fixed IEEE maximum of 1075 bits. These limits
are not Python-object/process-memory limits. Normalization uses 128 binary bits.
"""
from copy import deepcopy
from fractions import Fraction
import json
from math import isqrt
from types import MappingProxyType
import time

import numpy as np

SCHEMA='d92_support_metric_basis_v1'
ENCLOSURE_BITS=128


def _sealed_array(value):
    a=np.asarray(value,dtype=np.float64)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


def _freeze(value):
    if isinstance(value,dict):return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if isinstance(value,(list,tuple)):return tuple(_freeze(v) for v in value)
    return value


def _plain(value):
    if isinstance(value,MappingProxyType):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,tuple):return [_plain(v) for v in value]
    return deepcopy(value)


class _Limit(ArithmeticError):pass


class SupportMetricBasisFailure(RuntimeError):
    def __init__(self,code,audit,exact_state,Q):
        super().__init__(code);self.code=code;self._audit=_freeze(audit)
        self.exact_state=_freeze(exact_state);self.arrays=MappingProxyType({'Q':_sealed_array(Q)})
    def audit_dict(self):return _plain(self._audit)


class SupportMetricBasis:
    """Sealed attributes/arrays and immutable exact Fraction tuples.

    audit/audit_dict return independent mutable JSON-native copies. exact_basis
    contains unnormalized orthogonal residual columns, not irrational unit vectors.
    exact_zero_relations stores (original_column, coefficients), Q@coefficients=0.
    """
    __slots__=('Q','rank','column_indices','exact_basis','exact_residual_coefficients',
        'projection_coefficients','exact_zero_relations','exact_norm_squared','exact_unit_lower',
        'exact_unit_upper','basis','basis_lower','basis_upper','error_bounds',
        'spectral_error_bound','orthogonality_error_bound','_audit','_certificate','_sealed')
    def __init__(self,**values):
        for key,value in values.items():object.__setattr__(self,key,value)
        object.__setattr__(self,'_sealed',True)
    def __setattr__(self,key,value):raise AttributeError('SupportMetricBasis is sealed')
    @property
    def audit(self):return self.audit_dict()
    def audit_dict(self):return _plain(self._audit)
    def certificate_json(self):return self._certificate


class _Ledger:
    def __init__(self,bits,operations):
        self.bits=bits;self.operations=operations;self.started=time.perf_counter()
        self.data=dict(fraction_operations_attempted=0,fraction_operations_completed=0,
            fraction_operations_by_kind={},integer_operations_attempted=0,integer_operations_completed=0,
            integer_operations_by_kind={},integer_guard_checks=0,max_observed_integer_bits=0,
            max_intermediate_bit_bound=0,binary64_decodes=0,float_conversions=0,nextafter_steps=0,
            square_root_enclosures=0,columns_started=0,columns_completed=0)
    def observe(self,value):
        integers=(value.numerator,value.denominator) if isinstance(value,Fraction) else (value,)
        largest=max(abs(v).bit_length() for v in integers)
        self.data['max_observed_integer_bits']=max(self.data['max_observed_integer_bits'],largest)
        if largest>self.bits:raise _Limit('INTEGER_BIT_LIMIT')
    def guard(self,bits):
        self.data['integer_guard_checks']+=1
        self.data['max_intermediate_bit_bound']=max(self.data['max_intermediate_bit_bound'],bits)
        if bits>self.bits:raise _Limit('INTEGER_BIT_LIMIT')
    def f(self,kind,a,b=None):
        self.data['fraction_operations_attempted']+=1
        if self.data['fraction_operations_attempted']>self.operations:raise _Limit('FRACTION_OPERATION_LIMIT')
        a=Fraction(a);b=None if b is None else Fraction(b)
        an,ad=abs(a.numerator).bit_length(),a.denominator.bit_length()
        if b is not None:
            bn,bd=abs(b.numerator).bit_length(),b.denominator.bit_length()
            bound=(max(an+bd,bn+ad)+1 if kind in ('add','subtract') else
                max(an+bn,ad+bd) if kind=='multiply' else
                max(an+bd,ad+bn) if kind=='divide' else max(an+bd,bn+ad))
            self.guard(bound)
        else:self.guard(max(an,ad))
        if kind=='add':v=a+b
        elif kind=='subtract':v=a-b
        elif kind=='multiply':v=a*b
        elif kind=='divide':v=a/b
        elif kind=='less':v=a<b
        elif kind=='absolute':v=abs(a)
        elif kind=='is_zero':v=a==0
        else:raise AssertionError(kind)
        if not isinstance(v,bool):self.observe(v)
        self.data['fraction_operations_completed']+=1
        counts=self.data['fraction_operations_by_kind'];counts[kind]=counts.get(kind,0)+1
        return v
    def integer(self,kind,bound,call):
        self.data['integer_operations_attempted']+=1;self.guard(bound)
        value=call();self.observe(value)
        self.data['integer_operations_completed']+=1
        counts=self.data['integer_operations_by_kind'];counts[kind]=counts.get(kind,0)+1
        return value
    def dot(self,a,b):
        result=Fraction(0)
        for x,y in zip(a,b):result=self.f('add',result,self.f('multiply',x,y))
        return result
    def decode(self,value):
        self.data['binary64_decodes']+=1
        result=Fraction.from_float(float(value));self.observe(result);return result
    def outward(self,value,upper):
        self.data['float_conversions']+=1;f=float(value)
        if not np.isfinite(f):raise _Limit('FLOAT_ENCLOSURE_UNREPRESENTABLE')
        exact=self.decode(f)
        wrong=self.f('less',exact,value) if upper else self.f('less',value,exact)
        if wrong:
            f=float(np.nextafter(f,np.inf if upper else -np.inf));self.data['nextafter_steps']+=1
            if not np.isfinite(f):raise _Limit('FLOAT_ENCLOSURE_UNREPRESENTABLE')
            exact=self.decode(f)
        wrong=self.f('less',exact,value) if upper else self.f('less',value,exact)
        if wrong:raise _Limit('OUTWARD_ROUNDING_READBACK_FAILED')
        return f
    def sqrt_enclosure(self,S):
        """isqrt(floor(S*2**256)); endpoints enclose sqrt(S) exactly."""
        if S<0:raise ArithmeticError('NEGATIVE_EXACT_NORM')
        self.data['square_root_enclosures']+=1;b=ENCLOSURE_BITS
        shift=self.integer('left_shift',abs(S.numerator).bit_length()+2*b,lambda:S.numerator<<(2*b))
        floor=self.integer('floor_divide',max(1,shift.bit_length()),lambda:shift//S.denominator)
        m=self.integer('isqrt',max(1,(floor.bit_length()+1)//2),lambda:isqrt(floor))
        square=self.integer('multiply',max(1,2*m.bit_length()),lambda:m*m)
        scaled_square=self.integer('multiply',max(1,square.bit_length()+S.denominator.bit_length()),lambda:square*S.denominator)
        denominator=self.integer('left_shift',b+1,lambda:1<<b)
        high=m if scaled_square==shift else self.integer('add',m.bit_length()+1,lambda:m+1)
        low,high=Fraction(m,denominator),Fraction(high,denominator)
        self.observe(low);self.observe(high)
        return low,high
    def audit(self,**fields):
        return dict(self.data,**fields,wall_seconds=time.perf_counter()-self.started,
            max_integer_bits=self.bits,max_fraction_operations=self.operations,enclosure_bits=ENCLOSURE_BITS,
            operation_scope='Instrumented rational algebra/comparisons and enclosure integer arithmetic; loop/sign/shape/serialization control predicates and Fraction internal gcd/allocation counts are not measured',
            integer_limit_scope='Conservative cross-product/intermediate bounds before arithmetic; decoded binary64 ratios have a fixed maximum of 1075 bits',
            fraction_internal_integer_operations=None,process_peak_memory_bytes=None,spaceborne_cost=None)


def _exact_encode(value):
    if isinstance(value,Fraction):
        return {'numerator_hex':format(value.numerator,'x'),'denominator_hex':format(value.denominator,'x')}
    if isinstance(value,(tuple,list)):return [_exact_encode(v) for v in value]
    if isinstance(value,dict):return {k:_exact_encode(v) for k,v in value.items()}
    return value


def _integer_payload_bytes(value):
    if isinstance(value,Fraction):return (abs(value.numerator).bit_length()+7)//8+(value.denominator.bit_length()+7)//8
    if isinstance(value,(tuple,list)):return sum(_integer_payload_bytes(v) for v in value)
    if isinstance(value,dict):return sum(_integer_payload_bytes(v) for v in value.values())
    return 0


def build_support_metric_basis(*,Q,max_integer_bits,max_fraction_operations):
    """Exact fixed-order rank + enclosed normalized physical residuals.

    Accepted shape: 0..160 rows, 0..5 columns, finite float64. No tolerance
    deletes a column. Guard exhaustion is a technical failure with partial exact
    state, never evidence of a smaller rank. Explicit resource limits have no defaults.
    """
    for name,v in (('max_integer_bits',max_integer_bits),('max_fraction_operations',max_fraction_operations)):
        if type(v) is not int or v<=0:raise ValueError(name+' must be an explicit positive integer')
    raw=np.asarray(Q)
    if raw.dtype!=np.dtype('float64') or raw.ndim!=2 or raw.shape[0]>160 or raw.shape[1]>5 or not np.isfinite(raw).all():
        raise ValueError('Q must be a finite float64 matrix with at most 160 rows and 5 columns')
    raw=np.array(raw,copy=True);n,p=raw.shape;ledger=_Ledger(max_integer_bits,max_fraction_operations)
    state=dict(input_columns=[],orthogonal_residuals=[],residual_coefficients=[],projection_coefficients=[],
        norm_squared=[],column_indices=[],zero_relations=[],unit_lower=[],unit_upper=[])
    try:
        columns=[tuple(ledger.decode(v) for v in raw[:,j]) for j in range(p)];state['input_columns']=columns
        for j,q in enumerate(columns):
            ledger.data['columns_started']+=1;v=list(q);representation=[Fraction(int(i==j)) for i in range(p)];projections=[]
            for old,norm,old_rep in zip(state['orthogonal_residuals'],state['norm_squared'],state['residual_coefficients']):
                coefficient=ledger.f('divide',ledger.dot(old,q),norm);projections.append(coefficient)
                v=[ledger.f('subtract',x,ledger.f('multiply',coefficient,y)) for x,y in zip(v,old)]
                representation=[ledger.f('subtract',x,ledger.f('multiply',coefficient,y)) for x,y in zip(representation,old_rep)]
            state['projection_coefficients'].append(tuple(projections))
            zeros=[ledger.f('is_zero',x) for x in v]
            if all(zeros):state['zero_relations'].append((j,tuple(representation)))
            else:
                norm=ledger.dot(v,v)
                if norm<=0:raise ArithmeticError('NONPOSITIVE_EXACT_RESIDUAL_NORM')
                state['column_indices'].append(j);state['orthogonal_residuals'].append(tuple(v))
                state['residual_coefficients'].append(tuple(representation));state['norm_squared'].append(norm)
            ledger.data['columns_completed']+=1
        rank=len(state['column_indices']);approx=np.empty((n,rank));lower=np.empty_like(approx);upper=np.empty_like(approx);errors=np.empty_like(approx)
        maximum_error=Fraction(0);normalizations=[]
        for j,v in enumerate(state['orthogonal_residuals']):
            maximum=Fraction(0)
            for x in v:
                a=ledger.f('absolute',x)
                if ledger.f('less',maximum,a):maximum=a
            scaled=tuple(ledger.f('divide',x,maximum) for x in v)
            S=ledger.dot(scaled,scaled);lo,hi=ledger.sqrt_enclosure(S)
            if lo<=0:raise ArithmeticError('NORMALIZATION_LOWER_BOUND_NONPOSITIVE')
            lows=[];highs=[]
            for i,x in enumerate(scaled):
                negative=ledger.f('less',x,Fraction(0))
                l=ledger.f('divide',x,lo if negative else hi);h=ledger.f('divide',x,hi if negative else lo)
                midpoint=ledger.f('divide',ledger.f('add',l,h),Fraction(2))
                ledger.data['float_conversions']+=1;f=float(midpoint)
                if not np.isfinite(f):raise ArithmeticError('BASIS_UNREPRESENTABLE')
                actual=ledger.decode(f)
                left=ledger.f('absolute',ledger.f('subtract',actual,l));right=ledger.f('absolute',ledger.f('subtract',h,actual))
                error=right if ledger.f('less',left,right) else left
                if ledger.f('less',maximum_error,error):maximum_error=error
                approx[i,j]=f;lower[i,j]=ledger.outward(l,False);upper[i,j]=ledger.outward(h,True);errors[i,j]=ledger.outward(error,True)
                lows.append(l);highs.append(h)
            state['unit_lower'].append(tuple(lows));state['unit_upper'].append(tuple(highs))
            normalizations.append(dict(maximum_abs_residual=maximum,scaled_norm_squared=S,sqrt_lower=lo,sqrt_upper=hi))
        # ||E||_2 <= ||E||_F <= sqrt(n*r)*max_ij certified component error.
        squared=ledger.f('multiply',ledger.f('multiply',maximum_error,maximum_error),Fraction(n*rank))
        _,error_upper=ledger.sqrt_enclosure(squared)
        spectral_error=ledger.outward(error_upper,True)
        e=ledger.decode(spectral_error)
        orthogonal_error=ledger.f('add',ledger.f('multiply',Fraction(2),e),ledger.f('multiply',e,e))
        orthogonal_error=ledger.outward(orthogonal_error,True)
        certificate=dict(schema=SCHEMA,shape=[n,p],rank=rank,enclosure_bits=ENCLOSURE_BITS,
            input_scope='EXACT_STORED_BINARY64_NOT_PRE_ROUNDING_PROTOTYPES',fraction_encoding='signed_hex_numerator_positive_hex_denominator',
            exact_state=state,normalizations=normalizations)
        encoded=json.dumps(_exact_encode(certificate),ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':'))
        arrays=[_sealed_array(a) for a in (raw,approx,lower,upper,errors)]
        audit=ledger.audit(status='COMPLETE',exact_rank=rank,input_rows=n,input_columns=p,
            exact_zero_columns=len(state['zero_relations']),retained_nonzero_columns=rank,
            spectral_basis_error_upper=spectral_error,orthogonality_error_upper=orthogonal_error,
            numeric_array_payload_bytes=sum(a.nbytes for a in arrays),
            logical_rational_integer_payload_bytes=_integer_payload_bytes(certificate),certificate_utf8_bytes=len(encoded.encode('utf-8')),
            bytes_scope='Actual ndarray buffers and serialized data-only certificate bytes; rational integer bytes are logical repeated-value payload, excluding signs/type/object overhead',
            error_scope='Encloses the exact orthonormal basis of the stored float64 Q; no head/JVP/score/generalization certificate')
        return SupportMetricBasis(Q=arrays[0],rank=rank,column_indices=tuple(state['column_indices']),
            exact_basis=tuple(state['orthogonal_residuals']),exact_residual_coefficients=tuple(state['residual_coefficients']),
            projection_coefficients=tuple(state['projection_coefficients']),exact_zero_relations=tuple(state['zero_relations']),
            exact_norm_squared=tuple(state['norm_squared']),exact_unit_lower=tuple(state['unit_lower']),exact_unit_upper=tuple(state['unit_upper']),
            basis=arrays[1],basis_lower=arrays[2],basis_upper=arrays[3],error_bounds=arrays[4],
            spectral_error_bound=spectral_error,orthogonality_error_bound=orthogonal_error,_audit=_freeze(audit),_certificate=encoded)
    except Exception as exc:
        code=str(exc) if isinstance(exc,(_Limit,ArithmeticError)) else type(exc).__name__
        audit=ledger.audit(status='TECHNICAL_FAILURE',failure_code=code,exact_rank_complete=False,
            completed_independent_columns=len(state['column_indices']),completed_columns=ledger.data['columns_completed'],
            numeric_array_payload_bytes=raw.nbytes,logical_rational_integer_payload_bytes=_integer_payload_bytes(state),
            certificate_utf8_bytes=None,bytes_scope='Preserved input ndarray and completed exact partial state logical integer payload; failed certificate was not serialized')
        raise SupportMetricBasisFailure(code,audit,state,raw) from exc
