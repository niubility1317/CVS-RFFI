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
    t1=c.get('method_id')=='pamf_t1_e50'
    if read(c['source_contract']) != expected_contract or (not t1 and (done.get('initialization') != 'scratch' or done.get('ancestors') != [])):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    if t1:
        if done.get('initialization')!='same_contract_scratch20_own10_own20' or len(done.get('segments',[]))!=3:raise ValueError('T1 trajectory missing')
        previous=None
        for epochs,offset,s in zip((20,10,20),(0,20,30),done['segments']):
            cfg=s['config'];p=Path(s['folder'])
            if s['status']!='SOURCE_FROZEN' or s['updates']!=epochs*222 or s['epoch']!=epochs or s['target_access'] or cfg['parent_source']!=previous or cfg['step_offset']!=offset*222 or read(cfg['source_contract'])!=expected_contract:raise ValueError('T1 segment ancestry/roles differ')
            cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
            if cp['args']['baseline_ckpt']!=('' if previous is None else previous+'/final_ssdg.pth') or cp['args']['teacher_ckpt'] or cp['a1_ema_successful_updates']!=epochs*222:raise ValueError('T1 segment checkpoint differs')
            previous=str(p)
    payload = torch.load(folder / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    if payload.get('run_id') != c['run_id'] or payload.get('candidate_id') != c['row_id'] or payload.get('epoch') != (20 if t1 else 50) or payload.get('a1_ema_successful_updates') != (4440 if t1 else 11100):
        raise ValueError('F checkpoint belongs to a different source row')
    return reader_from_payload(c, payload, device), c


def load_original_F(seed, expected_contract, device, arm='T1'):
    if arm=='T1':return load_original_T1(seed,expected_contract,device)
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


def load_original_T1(seed,expected_contract,device):
    from experiments.cvs_f_supcon50 import design
    project=Path(design.PROJECT)
    run='20261020-phase1-T1-AsKL0-e50-s2026092701-m2-r01' if seed==2026092701 else '20261021-phase1-T1-AsKL0-e30plus20-m3-r01'
    folder=project/'runs'/run/('E30PLUS20-s%d'%seed)/'source'
    final=folder;chain=[]
    while True:
        done=read(folder/'completion.json');init=read(folder/'initialization.json');actual=read(folder/'source_contract.json');cfg=done['config']
        if any(actual[k]!=expected_contract[k] for k in ('role_ids','counts','source_rxs','source_days')) or cfg['model_seed']!=seed or done['target_access'] or init['target_contact']:
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        cp=torch.load(folder/'final_ssdg.pth',map_location='cpu',weights_only=False)
        if cp['run_id']!=cfg['run_id'] or cp['candidate_id']!=cfg['row_id'] or cp['epoch']!=cfg['epochs'] or cp['a1_ema_successful_updates']!=cfg['updates'] or cp['checkpoint_selection']!='final_only' or cp['args']['teacher_ckpt']:
            raise ValueError('T1 fixed source checkpoint identity differs')
        chain.append(cfg)
        if cfg['segment']==20:
            if cp['args']['baseline_ckpt'] or not init['scratch_only']:raise ValueError('T1 source ancestor is not scratch')
            break
        parent=Path(cfg['parent_source'])
        if cp['args']['baseline_ckpt']!=str(parent/'final_ssdg.pth') or init['checkpoint_sources']!=[str(parent/'final_ssdg.pth')]:raise ValueError('T1 own parent differs')
        folder=parent
    if [c['epochs'] for c in chain]!=[20,10,20] or sum(c['label_epochs'] for c in chain)!=13 or sum(c['pseudo_epochs'] for c in chain)!=37 or any(c['temperature']['As_KL']!=0 for c in chain):raise ValueError('Final T1 13L37U trajectory differs')
    payload=torch.load(final/'final_ssdg.pth',map_location='cpu',weights_only=False)
    c=dict(model_seed=seed,run_id=run,row_id='E30PLUS20-s%d'%seed,output_root=str(final),source_rxs=expected_contract['source_rxs'],source_days=expected_contract['source_days'],num_classes=6,regime='native_cric',satellite_concat=False,ablation=None)
    return reader_from_payload(c,payload,device)


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
