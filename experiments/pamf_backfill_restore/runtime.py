"""Use each original strict checkpoint loader; no training or source promotion."""
import importlib
from pathlib import Path
import numpy as np
import torch
from . import design as d
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import fixed_noise


def validate_training(value, candidate):
    expected=dict(candidate=candidate,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
                  TX_labels_read=0,target_reads=0)
    # R10 stores kind in place of candidate; all newer rows store candidate.
    stored=value.get('candidate',value.get('kind'))
    if stored is None and candidate=='learned_operator':stored=candidate  # R07 has one fixed model and no row field.
    if stored!=candidate or any(value.get(k)!=v for k,v in expected.items() if k!='candidate'):
        raise ValueError('Original completed full-LU source training required')
    if value.get('status') not in ('TRAINING_COMPLETE','SOURCE_TRAINED'):
        raise ValueError('Incomplete training cannot enter formal backfill')
    for key in ('F_outputs_read','F_weights_read'):
        if value.get(key,0)!=0:raise ValueError('This audit requires zero-F-training restorers')
    if value.get('training_F_access',False) or value.get('target_training',False):
        raise ValueError('Restorer training permissions differ')


def validate_final(payload, spec):
    if (payload.get('epoch'),payload.get('updates'))!=(20,4940):
        raise ValueError('Original final E20/4940 checkpoint required; no interim weights')
    candidate=payload.get('candidate',payload.get('kind'))
    if candidate is not None and candidate!=spec['candidate']:raise ValueError('Wrong checkpoint candidate')


class Adapter(torch.nn.Module):
    def __init__(self, model, spec, device):
        super().__init__();self.model=model.requires_grad_(False).eval();self.spec=spec
        number=int(spec['row_id'][1:3]);self.number=number
        self.generator=None
        if number in (20,21):
            design=importlib.import_module('experiments.'+spec['module'].split('.')[0]+'.design')
            self.generator=torch.Generator(device=device).manual_seed(design.PRIMARY_SEED)

    def forward(self, received, ids):
        name=self.spec['candidate'];module=self.spec['module'].split('.')[0]
        if self.number in (1,2,3) and ('diffusion' in name or 'flow' in name):
            design=importlib.import_module('experiments.'+module+'.design')
            return self.model(received,fixed_noise(ids,design.EVALUATION_SEED,received.shape,received.device))
        if self.generator is not None:
            noise=torch.randn(received.shape,dtype=received.dtype,device=received.device,generator=self.generator)
            return self.model(received,noise)
        return self.model(received)


def load(args, row_id, device):
    if row_id not in d.BY_ID:raise ValueError('Unregistered audit checkpoint')
    spec=d.BY_ID[row_id];root=args.runs_root/spec['run']
    if read_json(root/'launch.json')['commit']!=spec['commit']:raise ValueError('Original code provenance differs')
    validate_training(read_json(root/spec['training']),spec['candidate'])
    loader=importlib.import_module('experiments.'+spec['module']).load_checkpoint
    if int(row_id[1:3]) in (1,2,3):model,payload=loader(root/spec['checkpoint'],spec['candidate'],device)
    else:model,payload=loader(root/spec['checkpoint'],device)
    validate_final(payload,spec)
    return Adapter(model,spec,device).to(device).eval(),payload


def audit_contract(args):
    value=read_json(args.output/'resolved_config.json')
    if (value.get('authorization')!=d.AUTHORIZATION or value.get('rows')!=d.ROWS
            or value.get('new_training_updates')!=0 or value.get('target_feedback') is not False
            or value.get('source_decisions_modified') is not False or value.get('commit')!=args.expected_commit):
        raise ValueError('Fixed user-authorized audit matrix required')


def reference(args,role):
    if role!='target':raise ValueError('Formal audit is target-only')
    audit_contract(args)
    if args.row not in d.BY_ID:raise ValueError('Unregistered target row')
    manifest=read_json(args.formal_reference/'manifest.json')
    from experiments.pamf_uncertainty_restore.evaluation import VIEWS
    if (manifest.get('status')!='FORMAL_REFERENCE_READY' or manifest.get('count')!=168000
            or manifest.get('views')!=list(VIEWS) or manifest.get('original_IQ_bytes_reused') is not True
            or manifest.get('TX_truth_reads')!=0):
        raise ValueError('Original received formal reference required')
    ids=np.load(args.formal_reference/'physical_ids.npy',allow_pickle=False)
    if ids.shape!=(168000,) or len(np.unique(ids))!=168000:raise ValueError('Fixed unique formal physical IDs required')
    return args.formal_reference,ids,None,None
