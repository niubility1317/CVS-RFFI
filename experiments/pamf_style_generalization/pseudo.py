"""Reuse source-only pseudo identities; never load the historical refined model."""
import json
from pathlib import Path
import numpy as np
import torch
from . import design as d


def load_pseudo(root,parent_cache,source_cache):
    root=Path(root);launch=json.loads((root/'launch.json').read_text())
    bank=json.loads((root/'consensus_native/identity_bank_complete.json').read_text())
    feature=json.loads((root/'consensus_native/feature_cache_complete.json').read_text())
    argv=launch['argv']
    if (root.name!=d.PSEUDO_RUN or launch['commit']!=d.PSEUDO_COMMIT or
        any(Path(argv[argv.index('--'+name)+1]).resolve()!=Path(path).resolve()
            for name,path in (('parent-cache',parent_cache),('source-cache',source_cache))) or
        bank['source_records']!=63000 or bank['native_confidence']!=.9 or
        bank['consensus_accepted']!=59402 or bank['accepted_L_s']!=6074 or bank['accepted_U_s']!=53328 or
        any(bank.get(k)!=0 for k in ('TX_labels_read','V_reads','target_reads','head_fit_updates','F_optimizer_updates')) or
        feature['F_seeds']!=list(d.TEACHER_SEEDS) or feature['roles']!=['L_s','U_s'] or feature['count']!=63000):
        raise ValueError('Matching original source-only threeF pseudo artifact required')
    value=torch.load(root/'consensus_native/source_identity_bank.pt',map_location='cpu',weights_only=False)
    pseudo=value['pseudo'];accepted=value['accepted']
    if (pseudo.shape!=(63000,) or pseudo.dtype!=torch.long or accepted.shape!=(63000,) or accepted.dtype!=torch.bool or
        ((pseudo<0)|(pseudo>5)).any() or int(accepted.sum())!=59402):raise ValueError('Fixed source pseudo/mask layout required')
    return np.asarray(pseudo.tolist(),np.int64),np.asarray(accepted.tolist(),np.bool_)
