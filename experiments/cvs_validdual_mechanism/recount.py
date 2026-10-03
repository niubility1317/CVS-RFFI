"""Independent NumPy verification using orthonormal factor contrasts."""
import itertools
import numpy as np
from .analyze import OUTPUT,SOURCE,AXES,FACTORS,read,write
from experiments.cvs_validdual_clean.contracts import expected_rows

def helmert(n):
    b=np.zeros((n,n));b[:,0]=1/np.sqrt(n)
    for j in range(1,n):b[:j,j]=1/np.sqrt(j*(j+1));b[j,j]=-j/np.sqrt(j*(j+1))
    return b

def contrast_energies(x):
    q=np.einsum('ia,jb,kc,ijkd->abcd',helmert(6),helmert(5),helmert(3),x)
    result={}
    for bits in itertools.product((0,1),repeat=3):
        if not any(bits):continue
        indices=tuple(slice(1,None) if v else slice(0,1) for v in bits)
        result[':'.join(FACTORS[k] for k in range(3) if bits[k])]=float(np.sum(q[indices]**2)/90)
    return result

def relative(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(16,-1);b=np.asarray(b,dtype=np.float64).reshape(16,-1)
    return np.sqrt(((a-b)**2).sum(1))/np.maximum(np.sqrt((a*a).sum(1)),1e-12)

def recount():
    p=OUTPUT/'analysis';proof=p/'independent_recount.json'
    if proof.exists():raise FileExistsError('Preserve existing verification')
    done=read(p/'complete.json');factors=read(p/'source_factors.json')['rows'];rows=read(p/'public_metrics.json')['rows']
    if any(done.get(k)!=v for k,v in dict(status='ARTIFACTS_COMPLETE',rows=8,source_cells=720,public_pairs=512,target_access=False,new_checkpoints=False).items()) or len(factors)!=8 or len(rows)!=512:raise ValueError('Incomplete diagnostic')
    expected=set(expected_rows())
    if {r['row_id'] for r in factors}!=expected:raise ValueError('Factor matrix differs')
    keys={(r['row_id'],r['view'],r['public_index']) for r in rows}
    allkeys=set(itertools.product(expected,('phase','delay1','delay4','delay16'),range(16)))
    if keys!=allkeys or len(keys)!=len(rows):raise ValueError('Public matrix missing/duplicate')
    public=[dict(family=f,public_seed=s) for f in ('noise','qpsk','tone','periodic16') for s in range(91001,91005)]
    if read(p/'public_records.json')!=public:raise ValueError('Public signal labels differ')
    for row in [*factors,*rows]:
        variant,seed=expected_rows()[row['row_id']]
        if row.get('variant')!=variant or row.get('model_seed')!=seed:raise ValueError('Model row labels differ')
    for row in rows:
        if any(row.get(k)!=v for k,v in public[row['public_index']].items()):raise ValueError('Public row labels differ')
    inputs=np.load(p/'public_inputs.npz');continuous=inputs['continuous']
    for name in ('identity','phase','delay1','delay4','delay16'):
        if name=='identity':z=continuous[:,64:]
        elif name=='phase':z=continuous[:,64:]*np.exp(.4j)
        else:
            d=int(name[5:]);z=continuous[:,64:]+(.23+.09j)*continuous[:,64-d:320-d]
        z=z/np.sqrt(np.mean(abs(z)**2,axis=1,keepdims=True))
        reference=np.stack((z.real,z.imag),axis=1).astype(np.float32)
        if not np.array_equal(reference,inputs[name]):raise ValueError('Continuous public channel differs')
    errors=[]
    def compare(a,b):
        if not np.isfinite(a) or not np.isfinite(b) or not np.isclose(a,b,rtol=1e-10,atol=1e-12):raise ValueError('Independent numerical comparison failed')
        errors.append(abs(float(a)-float(b)))
    for factor in factors:
        rid=factor['row_id'];arrays=np.load(p/(rid+'.npz'));cells=read(SOURCE/rid/'source_final_diagnostics.json')['validdual_groups']
        x=np.empty((6,5,3,66));seen=set()
        if len(cells)!=90:raise ValueError('Independent source cell count differs')
        for cell in cells:
            key=tuple(cell[k] for k in ('tx','receiver','day'));a=np.asarray(cell['coefficient_mean']);v=cell['coefficient_trace_variance']
            if key in seen or any(k not in axis for axis,k in zip(AXES,key)) or cell['count']!=300:raise ValueError('Independent source balance/identity differs')
            if a.shape!=(66,) or not np.isfinite(a).all() or not np.isfinite(v) or v< -1e-12:raise ValueError('Independent source moments invalid')
            seen.add(key);x[tuple(axis.index(k) for axis,k in zip(AXES,key))]=a
        if not np.array_equal(x,arrays['cell_means']):raise ValueError('Source means not preserved')
        energies=contrast_energies(x)
        if set(factor['components'])!=set(energies) or set(factor['fractions_of_total'])!=set(energies):raise ValueError('Factor terms differ')
        for name,value in energies.items():compare(value,factor['components'][name])
        compare(sum(energies.values()),factor['between_cell_variance'])
        within=sum(max(0.,c['coefficient_trace_variance']) for c in cells)/90
        compare(within,factor['within_cell_variance']);total=within+sum(energies.values());compare(total,factor['total_variance'])
        if factor.get('numerical_floor')!=1e-12 or factor.get('degenerate') is not bool(total<=1e-12):raise ValueError('Degenerate/numerical floor differs')
        if total<=1e-12:
            if factor['within_fraction'] is not None:raise ValueError('Degenerate within fraction must be null')
        else:
            compare(within/total,factor['within_fraction'])
            compare(sum(factor['fractions_of_total'].values())+factor['within_fraction'],1.)
        for name,value in energies.items():
            if total<=1e-12:
                if factor['fractions_of_total'][name] is not None:raise ValueError('Degenerate fraction must be null')
            else:compare(value/total,factor['fractions_of_total'][name])
        for view in ('phase','delay1','delay4','delay16'):
            values={name:relative(arrays['identity_'+key],arrays[view+'_'+key]) for name,key in [('corrected_drift','corrected'),('raw_feature_drift','raw'),('fused_feature_drift','z')]}
            values['input_drift']=relative(inputs['identity'],inputs[view])
            # Main routine subtracts float32 tensors before FP64 norm: reproduce the stored inference precision explicitly.
            delta=(arrays['identity_coefficients']-arrays[view+'_coefficients']).astype(np.float64).reshape(16,-1)
            values['coefficient_change']=np.sqrt((delta*delta).sum(1))
            for row in [r for r in rows if r['row_id']==rid and r['view']==view]:
                i=row['public_index']
                for name,value in values.items():compare(value[i],row[name])
                for ratio,numerator,denominator in [('correction_ratio','corrected_drift','input_drift'),('feature_ratio','fused_feature_drift','raw_feature_drift')]:
                    if values[denominator][i]<=1e-6:
                        if row[ratio] is not None:raise ValueError('Numerical-floor ratio must be null')
                    else:compare(values[numerator][i]/values[denominator][i],row[ratio])
    write(proof,dict(status='VERIFIED',rows=8,source_cells=720,public_pairs=512,comparisons=len(errors),max_absolute_error=max(errors),independent_formula='orthonormal Helmert tensor contrasts plus direct FP64 sums',target_access=False))
    print('VERIFIED independent contrasts and all512 public pairs')

if __name__=='__main__':recount()
