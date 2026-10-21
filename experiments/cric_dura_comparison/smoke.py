"""One actual frozen F checkpoint and legal source IQ; never opens query."""
from pathlib import Path
from .io import read, write, tensor


def forward_source(net, x):
    import torch
    # The immutable native Sinc reads _version on computed physical-frequency
    # tensors. no_grad keeps those counters, unlike inference_mode.
    with torch.no_grad():
        return net(x)


def run(rt, handles):
    from .bootstrap import activate, numerical_policy
    activate(rt)
    numerical_policy()
    import numpy as np
    import torch
    from experiments.cvs_f_supcon50 import design, model, provenance
    c = design.config(dict(row_id='CROSS_RX-s2026092701', arm='CROSS_RX', model_seed=2026092701))
    source = Path(c['output_root'])
    contract = read(handles['F0']['source_contract'])
    original = read(source / 'source_contract.json')
    for key in ('role_ids', 'counts', 'source_rxs', 'source_days'):
        if original[key] != contract[key]:
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: ' + key)
    if original['num_classes'] != len(contract['classes']) or list(design.CLASSES) != contract['classes']:
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: class mapping')
    ck = torch.load(source / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    provenance.validate_checkpoint(ck, c)
    net = model.build(c, 'cpu').eval().requires_grad_(False)
    net.load_state_dict(ck['model'], strict=True)
    x = np.load(Path(handles['F0']['source_cache']) / 'L_s_x.npy', mmap_mode='r')[:2]
    out = forward_source(net, tensor(x))
    if out.shape != (2, 6) or not bool(torch.isfinite(out).all()):
        raise ValueError('Actual frozen F source smoke failed')
    write(Path(rt['run_root']) / 'real_checkpoint_smoke.json', dict(status='PASS', checkpoint=str(source / 'final_ssdg.pth'),
        samples=2, source_role='L_s', query_reads=0, U_TX_reads=0, head_fit_updates=0))
    print('REAL_CHECKPOINT_SOURCE_SMOKE_PASS', flush=True)
