"""Frozen SIG predictor diagnostics; no labels, fitting or model selection."""
import numpy as np

SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')


def errors(predicted,restored_signature,reference,iq,clean,basis):
    n=len(reference)
    if predicted.shape!=(2,n,160) or reference.shape!=(n,160) or restored_signature.shape!=reference.shape:
        raise ValueError('Two prior predictions and matched 160D source signatures required')
    if iq.shape!=clean.shape or iq.shape!=(n,2,256) or basis.ndim!=2 or basis.shape[0]!=160 or not 1<=basis.shape[1]<=5:
        raise ValueError('Matched source IQ and frozen-head contrast basis required')
    if not all(np.isfinite(v).all() for v in [predicted,restored_signature,reference,iq,clean,basis]):raise ValueError('Finite frozen source records required')
    p=predicted.astype(np.float64);t=reference.astype(np.float64);b=basis.astype(np.float64)
    if not np.allclose(b.T@b,np.eye(b.shape[1]),rtol=0,atol=2e-7):raise ValueError('Orthonormal frozen-head contrast basis required')
    energy=np.maximum((t*t).sum(1),1e-4);difference=p-t[None];visible=difference@b
    orthogonal=difference-visible@b.T
    output=restored_signature.astype(np.float64)-t;x=clean.astype(np.float64);r=iq.astype(np.float64)
    return dict(SIG_relative_MSE=(difference*difference).sum(2)/energy,
        SIG_head_visible_MSE=(visible*visible).sum(2)/energy,SIG_head_orthogonal_MSE=(orthogonal*orthogonal).sum(2)/energy,
        restored_SIG_relative_MSE=(output*output).sum(1)/energy,restored_SIG_head_visible_MSE=((output@b)**2).sum(1)/energy,
        iq_nmse=((r-x)**2).mean((1,2))/np.maximum((x*x).mean((1,2)),1e-4))


def summarize(values,mask):
    if not np.any(mask):raise ValueError('Nonempty registered source group required')
    result={}
    for key,array in values.items():
        if array.ndim==2:
            result[key]=float(array[:,mask].mean())
            for prior in [0,1]:result['prior'+str(prior)+'_'+key]=float(array[prior,mask].mean())
        else:result[key]=float(array[mask].mean())
    return result


def gain(before,after):return None if before==0 else 1-after/before


def decision(models):
    cells={}
    for name in ['linear','mlp']:
        train={v:{s:gain(models['context']['train'][v]['scene:'+s]['SIG_relative_MSE'],models[name]['train'][v]['scene:'+s]['SIG_relative_MSE']) for s in SCENES} for v in ['1','20']}
        validation={s:gain(models['context']['V'][s]['all']['SIG_relative_MSE'],models[name]['V'][s]['all']['SIG_relative_MSE']) for s in SCENES}
        visible={s:gain(models['context']['V'][s]['all']['SIG_head_visible_MSE'],models[name]['V'][s]['all']['SIG_head_visible_MSE']) for s in SCENES}
        train_counts={v:sum(x is not None and x>=.1 for x in values.values()) for v,values in train.items()}
        validation_no_gain=sum(x is not None and x<=0 for x in validation.values());validation_material=sum(x is not None and x>=.1 for x in validation.values())
        visible_no_gain=sum(x is not None and x<=0 for x in visible.values())
        cells[name]=dict(train_relative_gains=train,V_relative_gains=validation,V_head_visible_relative_gains=visible,
            train_scenes_atleast10percent_gain=train_counts,V_scenes_no_gain=validation_no_gain,V_scenes_atleast10percent_gain=validation_material,V_head_visible_scenes_no_gain=visible_no_gain,
            generalization_failure=all(x>=4 for x in train_counts.values()) and validation_no_gain>=4,
            head_direction_tradeoff=validation_material>=4 and visible_no_gain>=4)
    return dict(cells=cells,generalization_failure='PASS' if all(c['generalization_failure'] for c in cells.values()) else 'FAIL',
        head_direction_tradeoff='PASS' if all(c['head_direction_tradeoff'] for c in cells.values()) else 'FAIL',
        scope='Two separately fixed descriptive hypotheses; neither is an alternative primary, trueTX score, target benefit, model selection or automatic training authorization')
