"""Decompose FIR sensitivity without fitting, source IQ, query or new selection."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'code')]
import numpy as np
import torch
from torch.nn import functional as F
from experiments.cvs_mirror_subspace_identity.model import build, VARIANTS
from experiments.cvs_mirror_subspace_identity.collect import validate_completed, SOURCE_COMMIT
from experiments.cvs_mirror_subspace_identity.publish import CONNECTION, ssh
from experiments.cvs_spectral_relation_identity.physics import public_inputs, causal_fir, unit_rms

RUN = '20261003-diagnostic-cvs-mirror-mechanism-public-m8-r01'
SOURCE_RUN = '20261003-phase1-cvs-mirror-subspace-identity-manysig-m8-r01'
DELAYS = (2, 8, 16)
SEEDS = (2026092701, 2026092702, 2026092703, 2026092704)
GROUPS = ('noise', 'tone', 'periodic', 'dc', 'zero')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def validate_payload(payload, row):
    """Check the actual downloaded checkpoint against audited source provenance."""
    r = row['resolved']
    init = row['initialization']
    if (init.get('status') != 'SCRATCH' or init.get('ancestors') != [] or
            init.get('checkpoint_sources') != [] or init.get('physical_roles') != 'EXACT_MATCH' or
            init.get('target_access') is not False or init.get('target_contact') is not False or
            r.get('commit') != SOURCE_COMMIT):
        raise ValueError('Unverified source provenance')
    expected = dict(method='cvs_mirror_subspace_identity', variant=r['variant'], epoch=200,
                    num_classes=6, initialization=init, selection='fixed_last_epoch', config=r)
    if any(payload.get(k) != value for k, value in expected.items()):
        raise ValueError('Actual checkpoint metadata differs from audited source')
    contract = payload.get('source_contract', {})
    roles = contract.get('role_ids', {})
    if (payload.get('classes') != contract.get('classes') or len(payload.get('classes', [])) != 6 or
            {k: len(v) for k, v in roles.items()} != {'L_s': 6300, 'U_s': 56700, 'V': 27000}):
        raise ValueError('Actual checkpoint physical role counts differ')
    # Full contract equality is checked against independently downloaded source_contract.json by the caller.
    return contract


def load_verified_state(model, state, expected_contract):
    reference=model.state_dict()
    if set(state)!=set(reference):raise ValueError('Actual checkpoint state keys differ')
    for key,value in state.items():
        expected=reference[key]
        if (not isinstance(value,torch.Tensor) or value.layout!=torch.strided or
                value.dtype!=expected.dtype or value.shape!=expected.shape or not torch.isfinite(value).all()):
            raise ValueError('Actual checkpoint dtype/shape/finiteness differs: '+key)
    window='core.mirror_relation.window'
    if not torch.equal(state[window],reference[window]):raise ValueError('Actual periodic Hann window differs')
    model.load_state_dict(state,strict=True)
    if model.contract()!=expected_contract:raise ValueError('Actual loaded architecture contract differs')


@torch.no_grad()
def capture(block, x):
    v = block.mix_spectra(block.spectral(x))
    c = block.components(v)
    q = block.relations(v)
    z = torch.complex(v[:, 0].double(), v[:, 1].double())
    singular = torch.linalg.svdvals(z)
    return dict(q=q.cpu().numpy(), singular=singular.cpu().numpy(),
                energy_floor=c['energy_floor_active'].cpu().numpy(),
                determinant_floor=c['determinant_floor_active'].cpu().numpy(),
                determinant=c['determinant'].cpu().numpy(), alpha=c['alpha'].cpu().numpy())


@torch.no_grad()
def diagnose(model):
    model.eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    x = public_inputs(torch.device('cpu'), torch.float32)
    block = model.core.mirror_relation
    double_block = copy.deepcopy(block).double()
    arrays = {'iq': x.numpy()}
    for label, b, values in (('fp32', block, x), ('fp64', double_block, x.double())):
        for key, value in capture(b, values).items(): arrays[label+'_base_'+key] = value
        for delay in DELAYS:
            # Both dtypes start from identical float32 public samples; only diagnostic arithmetic changes.
            changed = unit_rms(causal_fir(values, [(0, 1., 0.), (delay, .23, .09)]))
            for key, value in capture(b, changed).items(): arrays[f'{label}_d{delay}_'+key] = value
    embedding = F.normalize(model.features(x), dim=1, eps=1e-4)
    arrays['embedding_base'] = embedding.numpy()
    for delay in DELAYS:
        changed = unit_rms(causal_fir(x, [(0, 1., 0.), (delay, .23, .09)]))
        arrays[f'embedding_d{delay}'] = F.normalize(model.features(changed), dim=1, eps=1e-4).numpy()
    if any(not torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise ValueError('Frozen public diagnostic mutated model state')
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite diagnostic output')
    return arrays


def run(root, output):
    spec_path = root/'automation_reports/CV-SincNet'/RUN/'experiment.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    if spec['status'] != 'LOCAL_VERIFIED' or len(spec['rows']) != 8:
        raise ValueError('Diagnostic must be preregistered once')
    if output.exists(): raise FileExistsError('Preserve existing output; inspect before any continuation')
    source_path = root/'local_artifacts/cvs_mirror_subspace_identity_20261003_r01/source_research_complete.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    validate_completed(source)
    rows = {r['row_id']: r for r in source['rows']}
    expected = {v+'-s'+str(s) for v in VARIANTS for s in SEEDS}
    if set(rows) != expected or {r['row_id'] for r in spec['rows']} != expected:
        raise ValueError('Fixed eight-row source matrix differs')
    identity = json.loads(ssh("import json,subprocess\nprint(json.dumps({k:subprocess.check_output([v],text=True).strip() for k,v in [('user','whoami'),('host','hostname')]}))"))
    if identity != {'user':'szu2070436088', 'host':'dell-DSS8440'}: raise ValueError('N607 identity mismatch')
    output.mkdir(parents=True)
    reference_path=output/'original_source_contract.json'
    reference_remote=next(iter(rows.values()))['resolved']['source_contract']
    subprocess.run(['scp',*CONNECTION,'N607:'+reference_remote,str(reference_path)],check=True,capture_output=True)
    reference_contract=json.loads(reference_path.read_text(encoding='utf-8'))
    if reference_contract['role_ids']['V'] != source['source_v_ids']:
        raise ValueError('Original source V IDs differ from the audited population')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    write(output/'launch.json', dict(run_id=RUN, pid=__import__('os').getpid(), commit=commit,
          source_commit=SOURCE_COMMIT, identity=identity, python=sys.executable, torch=torch.__version__,
          device='cpu', remote_writes=False, target_access=False, optimizer_steps=0))
    torch.set_num_threads(2)
    results = []
    for rid in sorted(rows):
        row=rows[rid]; folder=output/rid; folder.mkdir()
        checkpoint=row['completion']['checkpoint']; source_root=str(Path(checkpoint).parent).replace('\\','/')
        for name in ('last.pt','source_contract.json'):
            subprocess.run(['scp',*CONNECTION,'N607:'+source_root+'/'+name,str(folder/name)],check=True,capture_output=True)
        payload=torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
        actual_contract=json.loads((folder/'source_contract.json').read_text(encoding='utf-8'))
        if validate_payload(payload,row) != actual_contract or actual_contract != reference_contract:
            raise ValueError('Checkpoint and original source physical contract differ')
        model=build(row['resolved']['variant'])
        load_verified_state(model,payload['model'],row['resolved']['mirror_relation_actual'])
        started=time.time();arrays=diagnose(model);np.savez_compressed(folder/'public_arrays.npz',**arrays)
        result=dict(row_id=rid,variant=row['resolved']['variant'],model_seed=row['resolved']['model_seed'],
            checkpoint=checkpoint,source_commit=SOURCE_COMMIT,status='VERIFIED',public_packets=30,
            source_IQ_access=False,target_access=False,optimizer_steps=0,model_state_unchanged=True,
            runtime_seconds=time.time()-started,original_gpu_fir=row['physical_diagnostics']['finite_fir_sensitivity'])
        write(folder/'completion.json',result);results.append(result)
        print(json.dumps(result),flush=True)
    write(output/'completion.json',dict(status='VERIFIED',run_id=RUN,rows=results,training=False,target_access=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
