"""Keep restorer equivalence separate from descriptive frozen-reader differences."""
import torch


@torch.no_grad()
def verify_outputs(native,converted,reloaded):
    if any(len(values)!=4 for values in (native,converted,reloaded)):
        raise ValueError('Complete IQ/variance/features/logits tuple required')
    batch=native[0].shape[0]
    shapes=((batch,2,256),(batch,),(batch,160),(batch,6))
    for values in (native,converted,reloaded):
        if any(tuple(value.shape)!=shape or not torch.isfinite(value).all() for value,shape in zip(values,shapes)):
            raise ValueError('Registered finite IQ/variance/F shapes required')
        if (values[1]<=0).any():raise ValueError('Positive variance required')
    # The frozen restorer keeps the original tolerance. F is evaluated later on
    # every formal view; its nonlinear outputs are diagnostics, not a source gate.
    for a,b in zip(native[:2],converted[:2]):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    if any(not torch.equal(a,b) for a,b in zip(converted,reloaded)):
        raise ValueError('Strict frozen export reload output differs')
    records={}
    for key,a,b in zip(('IQ','variance','features','logits'),native,converted):
        try:torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6);close=True
        except AssertionError:close=False
        records[key]=dict(max_abs=float((a-b).abs().max()),
            relative_L2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-30)),
            elementwise_close=close)
    return dict(restorer_close=True,strict_reload_exact=True,outputs=records,
        argmax_disagreements=int((native[-1].argmax(1)!=converted[-1].argmax(1)).sum()),
        F_equivalence_gate=False,rule='IQ/variance keep original tolerance;alloutputs finite/strictreload;F differences descriptive;eachversion formal7')
