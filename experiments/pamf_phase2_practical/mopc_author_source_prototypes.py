"""Derive author-sampled source prototypes from a frozen, legal MoPC checkpoint."""
import argparse
from pathlib import Path

import torch

from .group1 import source_scope
from .group1_source import load_source, mopc_source_metadata, source_data
from .io import activate, read, require, write


def derive(runtime, output_root, seed):
    source_rt=source_scope(runtime)
    activate(runtime)
    torch.set_num_threads(2)
    seed=int(seed)
    original=Path(source_rt['run_root'])/'source'/f'mopc_hr-s{seed}'/'source.pt'
    root=Path(output_root)
    require(not root.exists(), 'Derived source output already exists')
    bundle=load_source(source_rt,'mopc_hr',seed,'cpu')
    source=source_data(source_rt,'L_s')
    labels=torch.tensor(source.metadata['y'].tolist(),dtype=torch.long)
    require(len(source.x)==len(labels), 'Source labels do not match source IQ')
    metadata=mopc_source_metadata(bundle['model'],source.x,labels,'cpu',seed=seed)
    payload=torch.load(original,map_location='cpu',weights_only=False)
    require(payload.get('seed')==seed and payload.get('target_contact') is False,
            'Unexpected original source provenance')
    payload['base_metadata']=metadata
    payload['source_prototype_derivation']=dict(parent_checkpoint=str(original),
        source_role='L_s',target_read=False,query_read=False,
        source_sampling=metadata['source_sampling'],source_sampling_seed=seed,
        source_rows=metadata['source_rows'],source_rows_used=metadata['source_rows_used'])
    folder=root/'source'/f'mopc_hr-s{seed}'
    folder.mkdir(parents=True,exist_ok=False)
    torch.save(payload,folder/'source.pt')
    write(root/'complete.json',dict(status='SOURCE_PROTOTYPES_DERIVED',seed=seed,
          source_checkpoint=str(folder/'source.pt'),parent_checkpoint=str(original),
          source_rows=metadata['source_rows'],source_rows_used=metadata['source_rows_used'],
          source_sampling=metadata['source_sampling'],target_read=False,query_read=False))
    print('AUTHOR_MOPC_SOURCE_PROTOTYPES_DERIVED',seed,metadata['source_rows_used'],flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',required=True)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--seed',required=True,type=int)
    args=parser.parse_args()
    derive(read(args.runtime),args.output_root,args.seed)


if __name__=='__main__':
    main()
