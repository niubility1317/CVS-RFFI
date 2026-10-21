"""Frozen joint-head F and exact-contract DURA readers; no fitted parameters."""
from pathlib import Path
from types import MethodType
import torch
from .io import read


def frozen_filters(self, device, dtype):
    from experiments.cvs_rdrop_sinc_units.coordinate import CoordinateSinc
    # Same arithmetic as native; retain normal versioned tensors for its cache.
    with torch.inference_mode(False), torch.no_grad():
        return CoordinateSinc._filters(self, device, dtype)


def reader_from_payload(c, payload, device):
    from .f_train import arguments, original_config, class_dimension
    from .f_model import install
    from experiments.cvs_phase1_stack.runtime import installed
    with class_dimension(c), installed(original_config(c['model_seed'])) as native:
        args=arguments(c,device);args.input_len=256
        net = install(native.build_baseline_model(args, device), c)
    net.load_state_dict(payload['model'], strict=True)
    reader = net.id_backbone.encoder
    sinc = reader.core.id_backbone.sinc
    sinc._filters = MethodType(frozen_filters, sinc)
    return reader.to(device).eval().requires_grad_(False)


def load_F(folder, expected_contract, device):
    folder = Path(folder)
    done = read(folder / 'complete.json')
    if done.get('status') != 'SOURCE_FROZEN' or done.get('updates') != 11100 or done.get('epoch') != 50 or done.get('target_access') is not False:
        raise ValueError('Complete source-only fixed-final F required')
    c = done['config']
    if read(c['source_contract']) != expected_contract or done.get('initialization') != 'scratch' or done.get('ancestors') != []:
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    payload = torch.load(folder / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    if payload.get('run_id') != c['run_id'] or payload.get('candidate_id') != c['row_id'] or payload.get('epoch') != 50 or payload.get('a1_ema_successful_updates') != 11100:
        raise ValueError('F checkpoint belongs to a different source row')
    return reader_from_payload(c, payload, device), c


def load_original_F(seed, expected_contract, device, arm='CROSS_RX'):
    from experiments.cvs_f_supcon50 import design, model
    c = design.config(dict(row_id=arm+'-s%d'%seed, arm=arm, model_seed=seed))
    folder = Path(c['baseline_reference']['source'] if arm=='BASE' else c['output_root'])
    actual = read(folder / 'source_contract.json')
    if any(actual[k] != expected_contract[k] for k in ('role_ids','counts','source_rxs','source_days')) or list(design.CLASSES)!=expected_contract['classes']:
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    payload = torch.load(folder/'final_ssdg.pth',map_location='cpu',weights_only=False)
    source_cfg=design.base_config(seed) if arm=='BASE' else c
    if payload.get('run_id')!=source_cfg['run_id'] or payload.get('candidate_id')!=source_cfg['row_id']:
        raise ValueError('Original F checkpoint identity differs')
    if payload.get('epoch')!=50 or payload.get('a1_ema_successful_updates')!=11100 or payload.get('checkpoint_selection')!='final_only':
        raise ValueError('Original fixed final50 F required')
    net=model.build(c,device) if arm!='BASE' else None
    if net is None:
        from experiments.cvs_rdrop_sinc_units import model as af_model
        net=af_model.build(design.base_config(seed),device)
    net.load_state_dict(payload['model'],strict=True)
    reader=net.id_backbone.encoder
    reader.core.id_backbone.sinc._filters=MethodType(frozen_filters,reader.core.id_backbone.sinc)
    return reader.eval().requires_grad_(False)


def load_G(spec, expected_contract, device):
    if spec.get('kind') == 'G_ablation':
        from .g_ablation import build
        from .g_lineage import skeleton
        payload=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
        if payload.get('schema')!='cric_dura_G_ablation_v1' or payload.get('source_contract')!=expected_contract or payload.get('epoch')!=20 or payload.get('updates')!=4940:
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        if any(payload.get(k)!=0 for k in ('TX_labels_read','F_reads','target_reads')) or payload['config']['arm']!=spec['arm']:
            raise ValueError('G ablation source permissions/arm differ')
        model=build(skeleton('r65'),spec['arm']);model.load_state_dict(payload['model'],strict=True)
        model._cd_eager_variant=True
        return model.to(device).eval().requires_grad_(False)
    if spec.get('kind') == 'contract_matched':
        from .g_lineage import load
        model,payload=load(spec['checkpoint'],expected_contract,spec['roles'],device)
        if payload['stage']!='r69':raise ValueError('Final two-stage DURA required')
    elif spec.get('kind') == 'original_F0_LU':
        if spec['roles']!='LU':raise ValueError('Original G has seen source U')
        actual=read(spec['source_contract'])
        if any(actual[k]!=expected_contract[k] for k in ('role_ids','counts','source_rxs','source_days')):
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        from experiments.pamf_r65_four_route.runtime import load_checkpoint
        model,payload=load_checkpoint(spec['checkpoint'],device)
        if payload['arm']!='stage2' or payload['epoch']!=20 or payload['updates']!=4940:
            raise ValueError('Original final DURA v1 required')
    else:raise ValueError('Registered G source kind required')
    if sum(p.numel() for p in model.parameters())!=266951:raise ValueError('Full DURA v1 parameters differ')
    return model.eval().requires_grad_(False)
