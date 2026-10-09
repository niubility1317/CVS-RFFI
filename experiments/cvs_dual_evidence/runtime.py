from contextlib import contextmanager
import json
import torch
from experiments.cvs_phase1_stack.runtime import installed as native_installed
from experiments.cvs_dual_evidence.model import upgrade


@contextmanager
def installed(c):
    with native_installed(c) as n:
        build=n.build_baseline_model;writer=n._write_ssdg_epoch_telemetry;state={}
        def dual_build(args,device):
            model=upgrade(build(args,device),c['arm'],c['model_seed']);state['model']=model
            return model
        def annotate(rows):
            model=state.get('model')
            if model is not None and rows:
                values=model.dual_diagnostics() if hasattr(model,'dual_diagnostics') else {}
                values.update(arm=c['arm'],parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    gradient_present_parameters=sum(p.numel() for p in model.parameters() if p.grad is not None),
                    parameter_buffer_bytes=sum(v.numel()*v.element_size() for v in list(model.parameters())+list(model.buffers())),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if next(model.parameters()).is_cuda else None,
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if next(model.parameters()).is_cuda else None,
                    metric_scope='last forward batch at epoch end; gradients from last training update')
                rows[-1].update({'dual_'+k:v for k,v in values.items()})
                print('DUAL '+json.dumps(dict(epoch=rows[-1]['epoch'],**values)),flush=True)
        def telemetry(cp,jp,rows):
            annotate(rows)
            return writer(cp,jp,rows)
        previous=getattr(n,'_dual_evidence_annotate',None)
        n._dual_evidence_annotate=annotate
        n.build_baseline_model=dual_build
        n._write_ssdg_epoch_telemetry=telemetry
        try:yield n
        finally:
            n.build_baseline_model=build;n._write_ssdg_epoch_telemetry=writer
            if previous is None:del n._dual_evidence_annotate
            else:n._dual_evidence_annotate=previous
