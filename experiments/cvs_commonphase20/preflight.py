"""One real matching source checkpoint smoke; no initialization or query use."""
import torch
from . import design as d,model as api

def smoke():
    torch.set_num_threads(2);source=d.SMOKE_SOURCE
    done=d.read(source/'completion.json');contract=d.read(source/'source_contract.json');fold=d.read(source/'source_fold.json');carrier=d.read(d.SOURCE)
    if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done.get('target_access') is not False:raise ValueError('Real smoke checkpoint source provenance differs')
    if contract.get('held_RX')!=1 or set(contract['source_receivers'])!=set(d.FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600} or fold['held_RX']!=1:raise ValueError('Real checkpoint effective fourRX contract differs')
    if any(not set(v)<=set(carrier['role_ids'][k]) for k,v in contract['role_ids'].items()):raise ValueError('Real checkpoint roles escape legal carrier')
    ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
    if ck['args'].get('source_fold_held_rx')!=1 or ck['args']['baseline_ckpt'] or ck['args']['teacher_ckpt']:raise ValueError('Real checkpoint inheritance mismatch')
    m=api.build(d.config(d.rows()[0]),'cpu').eval();m.load_state_dict(ck['model'],strict=True)
    g=torch.Generator().manual_seed(20261016204);x=torch.randn(2,2,256,generator=g)
    with torch.no_grad():z=m.forward_identity_only(x)['tx_logits']
    if z.shape!=(2,6) or not bool(torch.isfinite(z).all()) or m._supcon_handles or m._supcon_pending is not None:raise ValueError('Real checkpoint no-query smoke failed')
    result=dict(status='PASS',role='Real checkpoint loaded only for no-query compatibility smoke; never training initialization',checkpoint_source=str(source),same_effective_LUV_contract=True,strict_load=True,synthetic_NN_forward_packets=2,source_IQ_reads=0,target_access=False,training_updates=0,head_fit_updates=0)
    d.write(d.BASE/'real_checkpoint_smoke.json',result);return result
