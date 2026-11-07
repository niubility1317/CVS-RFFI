"""Compare dense and low-rank source posteriors at a reproduced guard."""
import torch

from . import response_epoch10_diagnostic as replay
from . import response_joint_prior_trial as science
from .cost_probe import write
from .mismatched_prior_training import choose_prior

RUN='20261108-diagnostic-Restore-ResponseE10Dense-source-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.response_epoch10_dense_diagnostic'
original_readout=science.response_readout


def summary(low,low_aux,dense,dense_aux):
    low_residual=low_aux['relative_solve_residual']
    dense_residual=dense_aux['relative_solve_residual']
    if not all(torch.isfinite(v).all() for v in (low,dense,low_residual,dense_residual)):
        raise ValueError('Finite compared posterior outputs required')
    return dict(low_rank_max_residual=float(low_residual.max()),dense_max_residual=float(dense_residual.max()),
                max_abs_iq_difference=float((low-dense).abs().max()),
                low_rank_over_limit=int((low_residual>replay.LIMIT).sum()),
                dense_over_limit=int((dense_residual>replay.LIMIT).sum()))


def main():
    replay.RUN=RUN;replay.MODULE=MODULE
    def instrumented(model,reader,received,clean,features,scale,objective,**kwargs):
        output=original_readout(model,reader,received,clean,features,scale,objective,**kwargs)
        if output[1]['relative_solve_residual'].max()>replay.LIMIT:
            try:
                with torch.no_grad():
                    indices=kwargs['source_indices'];epoch=kwargs['epoch'];trace=kwargs['source_trace']
                    prior=choose_prior(kwargs['priors'],kwargs['assignment'],indices,epoch)
                    delta,_=science.selected_perturbation(clean,indices,epoch)
                    h,main_path,image_path,_=science.known_parameters(trace)
                    changed=science.inject(received,delta,science.dense_action(h,main_path,image_path))
                    result={}
                    for name,value in (('original',received),('perturbed',changed)):
                        low,low_aux=model.source_readout(value,prior=prior)
                        dense,dense_aux=model._readout(value,prior)
                        result[name]=summary(low,low_aux,dense,dense_aux)
                    report=dict(status='COMPARISON_COMPLETE',epoch=epoch,microbatch=len(indices),
                        original_and_perturbed=result,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0)
            except Exception as error:
                report=dict(status='COMPARISON_FAILED',exception_type=type(error).__name__,exception_message=str(error)[:500])
            write(replay_output/'dense_comparison.json',report)
        return output
    import sys
    from pathlib import Path
    root=Path(sys.argv[sys.argv.index('--root')+1]);replay_output=root/'runs'/RUN
    science.response_readout=instrumented
    replay.main()


if __name__=='__main__':main()
