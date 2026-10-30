"""Describe existing paired source outputs; no models, fitting, or prediction."""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def case_statistics(clean, restored, agreement, rx):
    x=clean[:,0].astype(np.float64)+1j*clean[:,1].astype(np.float64)
    r=restored[:,0].astype(np.float64)+1j*restored[:,1].astype(np.float64)
    reference=np.fft.fft(x,norm='ortho');difference=np.fft.fft(r-x,norm='ortho')
    power=np.abs(reference)**2;error=np.abs(difference)**2;energy=power.sum(1)
    if not np.isfinite(error).all() or not (energy>0).all():raise ValueError('Finite nonzero source references required')
    order=np.argsort(-power,axis=1,kind='stable')
    q_power=np.stack([np.take_along_axis(power,order[:,q*64:(q+1)*64],axis=1).sum(1) for q in range(4)],1)
    q_error=np.stack([np.take_along_axis(error,order[:,q*64:(q+1)*64],axis=1).sum(1) for q in range(4)],1)
    nmse=error.sum(1)/energy
    np.testing.assert_allclose(error.sum(1),(np.abs(r-x)**2).sum(1),atol=1e-10,rtol=2e-12)
    np.testing.assert_allclose(q_power.sum(1),energy,atol=1e-10,rtol=2e-12)
    np.testing.assert_allclose(q_error.sum(1),error.sum(1),atol=1e-10,rtol=2e-12)
    masks={'all':np.ones(len(x),bool),'F_teacher_agrees':agreement.astype(bool),'F_teacher_differs':~agreement.astype(bool)}
    masks.update({f'rx:{value}':rx==value for value in np.unique(rx)})
    result={}
    for name,mask in masks.items():
        if not mask.any():result[name]=dict(count=0,statistics=None);continue
        total_error=float(q_error[mask].sum());total_clean=float(q_power[mask].sum())
        result[name]=dict(count=int(mask.sum()),mean_iq_nmse=float(nmse[mask].mean()),
            clean_energy_share=(q_power[mask].sum(0)/total_clean).tolist(),
            error_energy_share=(q_error[mask].sum(0)/total_error).tolist() if total_error>0 else None,
            whole_iq_nmse_contributions=(q_error[mask]/energy[mask,None]).mean(0).tolist(),
            quartile_relative_nmse=[float(q_error[mask,q].sum()/q_power[mask,q].sum()) if q_power[mask,q].sum()>0 else None for q in range(4)])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve existing analysis output')
    started=time.perf_counter();results={}
    for model in ('noVar','DURA_default'):
        results[model]={}
        for view in ('clean','1','20'):
            with np.load(args.artifacts/f'{model}-{view}.npz',allow_pickle=False) as values:
                assert values['clean'].shape==(512,2,256)
                results[model][view]={case:case_statistics(values['clean'],values[case],values[case+'/teacher_agreement'],values['rx']) for case in ('native','aligned')}
    value=dict(status='POSTHOC_EXISTING_SOURCE_OUTPUTS',results=results,source_run=args.artifacts.name,
        unique_source_records=512,model_view_cells=6,frequency_partition='Per-record cleanFFT energy rank;4equal64bin quartiles,strongest first;not fixed frequency bands',
        group_definition='OriginalsameFclean teacher-choice agreement;not TX accuracy',
        Parseval_and_partition_checks=True,new_IQ_inputs=0,new_model_forwards=0,new_F_forwards=0,optimizer_updates=0,
        TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started,
        scope='Descriptive source residual energy only;quartile errors/correlations do not identify TX traits or justify a loss by themselves')
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({m:{v:{c:results[m][v][c]['all'] for c in ('native','aligned')} for v in ('1','20')} for m in results},indent=2))


if __name__=='__main__':main()
