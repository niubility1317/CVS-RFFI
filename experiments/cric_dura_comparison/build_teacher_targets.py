"""One source-only producer for explicitly registered same-teacher consumers."""
import argparse
from pathlib import Path
from .io import read,write
from .teacher_targets import binding,produce_shared_epoch


def run(c,device='cuda:0'):
    if any(c.get(k) for k in ('truth','target_truth','target_inputs','tests','reader','F_checkpoint')):
        raise ValueError('Teacher cache producer rejects F/target inputs')
    from .g_cache import CacheView
    from .g_lineage import load
    view=CacheView(c['cache']);contract=read(c['cache']['source_contract']);roles=c['cache']['roles']
    if not view.ready():raise ValueError('Incomplete source observations')
    if c.get('legacy_R65',False):
        from experiments.pamf_r46_delayed_prune.runtime import load_checkpoint
        old=read(c['teacher_source_contract'])
        if roles!='LU' or any(old[k]!=contract[k] for k in ('role_ids','counts','source_rxs','source_days')):
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        teacher,payload=load_checkpoint(c['teacher_checkpoint'],device)
        if payload['arm']!='prune_after10' or payload['epoch']!=20 or payload['updates']!=4940:
            raise ValueError('Original fixed-final R65 required')
    else:
        teacher,_=load(c['teacher_checkpoint'],contract,roles,device)
    teacher.eval().requires_grad_(False)
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    write(root/'config.json',c)
    identity=binding(c['cache'],c['teacher_checkpoint'],c['teacher_stage'])
    for epoch in range(1,21):
        produce_shared_epoch(root,identity,view,teacher,epoch,positions=tuple(c.get('positions',[-2,-1])),batch_size=c.get('batch_size',64))
        print('TEACHER_TARGET_EPOCH_COMPLETE',epoch,flush=True)
    write(root/'complete.json',dict(status='ALL_TEACHER_TARGETS_COMPLETE',epochs=20,binding=identity,
        TX_labels_read=0,F_reads=0,target_reads=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    from .bootstrap import activate,numerical_policy
    c=read(args.config);activate(c);numerical_policy();run(c)
