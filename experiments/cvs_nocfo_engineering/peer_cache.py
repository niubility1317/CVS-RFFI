"""Strict fixed original-E20 teacher probability cache; never constructs a model."""
import json
from pathlib import Path
import torch

SCHEMA='nocfo_ce_peer_probability_cache_v1'
SEEDS=(2026092701,2026092702,2026092703)
COUNT=56700


def _proof(value,seed):
    from .teacher import check_proof
    check_proof(value,seed)


def load(c,U):
    teacher=c['teacher']
    if teacher['source']!='peer':return None
    if 'y' in U:raise ValueError('U truth forbidden')
    if U.get('role')!='U_s':raise ValueError('Expected physical source U_s cache')
    if c['model_seed'] not in SEEDS or teacher['temperature']!=1. or teacher['confidence']!=.95:raise ValueError('Teacher definition differs')
    wanted=[s for s in SEEDS if s!=c['model_seed']]
    if teacher['seeds']!=wanted:raise ValueError('Exactly the other two original seeds required')
    ids=list(U['ids'])
    if len(ids)!=COUNT or len(set(ids))!=COUNT:raise ValueError('U physical pool must contain 56700 unique PID')
    root=Path(teacher['cache_root']);completion=json.loads((root/'completion.json').read_text(encoding='utf-8'))
    expected=dict(schema=SCHEMA,status='COMPLETE',source_role='U_s',count=COUNT,seeds=list(SEEDS),target_access=False,U_truth=False)
    if any(completion.get(k)!=v for k,v in expected.items()):raise ValueError('Incomplete or contaminated teacher cache')
    if set(completion['provenance'])!={str(s) for s in SEEDS}:raise ValueError('Incomplete cache source proofs')
    for s in SEEDS:_proof(completion['provenance'][str(s)],s)
    probabilities=[];proofs={};paths=[]
    for seed in wanted:
        path=root/f'seed-{seed}.pt';value=torch.load(path,map_location='cpu',weights_only=False)
        if set(value)!={'schema','seed','source_role','ids','probabilities','provenance'}:raise ValueError('Unexpected cache fields; labels/extra states forbidden')
        if value['schema']!=SCHEMA or value['seed']!=seed or value['source_role']!='U_s':raise ValueError('Wrong teacher cache identity')
        _proof(value['provenance'],seed)
        if value['provenance']!=completion['provenance'][str(seed)]:raise ValueError('Cache/completion source proof mismatch')
        source_ids=list(value['ids'])
        if len(source_ids)!=COUNT or len(set(source_ids))!=COUNT or set(source_ids)!=set(ids):raise ValueError('Teacher physical PID set differs')
        q=value['probabilities']
        if not torch.is_tensor(q) or q.dtype!=torch.float32 or q.shape!=(COUNT,6) or q.requires_grad:raise ValueError('Expected detached float32 teacher probabilities')
        if not torch.isfinite(q).all() or (q<0).any() or (q>1).any() or not torch.allclose(q.sum(1),torch.ones(COUNT),atol=1e-6,rtol=1e-6):raise ValueError('Invalid teacher probability simplex')
        index={pid:i for i,pid in enumerate(source_ids)}
        order=torch.tensor([index[pid] for pid in ids],dtype=torch.long)
        probabilities.append(q[order].detach().clone());proofs[str(seed)]=value['provenance'];paths.append(str(path))
    left,right=probabilities;q=(left+right)/2
    # Exact probability mean is the teacher; clamping is only for log metrics.
    logq=q.clamp_min(1e-12).log()
    js=.5*((left*(left.clamp_min(1e-12).log()-logq)).sum(1)+(right*(right.clamp_min(1e-12).log()-logq)).sum(1))
    return dict(q=q,peer_disagreement=left.argmax(1)!=right.argmax(1),peer_JS=js,
        teacher_entropy=-(q*logq).sum(1),ids=ids,teacher_seeds=wanted,provenance=proofs,
        cache_files=paths,source_role='U_s',schema=SCHEMA,teacher_forward_per_step=0)
