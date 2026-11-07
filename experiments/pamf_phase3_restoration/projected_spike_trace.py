"""Instrument the same bounded source replay when a projected correction spikes."""
import sys
from pathlib import Path

import torch

from . import projected_epoch11_replay as base

RUN='20261108-diagnostic-Restore-ProjectedSpikeTrace-sourceLU-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.projected_spike_trace'


def tensor_stats(value):
    value=value.detach().double()
    if not value.numel() or not torch.isfinite(value).all():raise ValueError('Finite nonempty diagnostic tensor required')
    return dict(norm=float(value.norm()),rms=float(value.square().mean().sqrt()),max_abs=float(value.abs().max()))


def main():
    root=Path(sys.argv[sys.argv.index('--root')+1]);out=root/'runs'/RUN
    previous=base.original_modules
    def modules(path):
        model_api,items=previous(path);original=model_api.shrinkage_training_readout
        writer=items['cost_probe'].write
        def instrument(model,reader,received,clean,features,scale,objective):
            modules=[('stem',model.detail.stem),
                     *[('encoder_'+str(i),v) for i,v in enumerate(model.detail.encoders)],
                     *[('attention_'+str(i),v) for i,v in enumerate(model.detail.bottleneck)],
                     *[('decoder_'+str(i),v) for i,v in enumerate(model.detail.decoders)],
                     ('output',model.detail.output),('full_detail',model.detail)]
            activations={}
            def capture(name):
                def save(module,inputs,value):activations[name]=value.detach()
                return save
            hooks=[module.register_forward_hook(capture(name)) for name,module in modules]
            try:result=original(model,reader,received,clean,features,scale,objective)
            finally:
                for hook in hooks:hook.remove()
            restored,aux,loss,terms=result;correction=terms['projected_prior_update_nmse'];flagged=base.spike_indices(correction)
            if len(flagged):
                i=int(flagged[correction[flagged].argmax()]);detail=model.detail
                values=dict(status='SPIKE_TRACE_COMPLETE',microbatch=len(received),flagged_count=len(flagged),
                    selected_projected_nmse=float(correction[i]),selected_base_nmse=float(((aux['base_iq'][i]-clean[i]).square().mean()/clean[i].square().mean().clamp_min(1e-4))),
                    selected_predicted_noise=float(aux['variance'][i]),received=tensor_stats(received[i]),clean=tensor_stats(clean[i]),
                    posterior_base=tensor_stats(aux['base_iq'][i]),projected_update=tensor_stats(aux['projected_prior_update'][i]),
                    learned_proposal=tensor_stats(activations['full_detail'][i]-aux['base_iq'][i]),detail_output_weight=tensor_stats(detail.output.weight),
                    layers={name:tensor_stats(value[i]) for name,value in activations.items()},
                    TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,
                    limitations='Same bounded source replay; saved layer statistics describe only the largest flagged record in the first triggered microbatch.')
                writer(out/'spike_trace.json',values)
                print('PROJECTED_SPIKE_LAYER_TRACE_SAVED',flush=True)
            return result
        model_api.shrinkage_training_readout=instrument
        return model_api,items
    base.original_modules=modules;base.RUN=RUN;base.MODULE=MODULE
    base.main()


if __name__=='__main__':main()
