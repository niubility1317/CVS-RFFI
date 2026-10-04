"""Narrow native adaptations: chosen identity, fixed budget, blind U labels."""
from contextlib import contextmanager
from copy import deepcopy
import torch
from experiments.cvs_phase1_overlay.model import native_modules,Phase1IdentityAdapter


def blind_sample(sample):
    x,_,domain,raw=sample[:4]
    meta={k:v for k,v in dict(raw).items() if k not in {'tx','tx_i','true_tx_i','tx_label','y_tx'}}
    meta['tx_label_visible']=False
    return x,-1,domain,meta


@contextmanager
def installed(c):
    n=native_modules()
    originals={name:getattr(n,name) for name in ['build_baseline_model','WiSigSubsetDataset','_resolve_unlabeled_batch_size','_unlabeled_drop_last','_muse_epoch_pairs','resolve_device']}
    class HiddenU(originals['WiSigSubsetDataset']):
        def __init__(self,*a,**kw):
            self.hide='unlabeled' in kw.get('split_source','')
            super().__init__(*a,**kw)
        def __getitem__(self,i):
            s=super().__getitem__(i)
            return blind_sample(s) if self.hide else s
    def build(a,device):
        m=originals['build_baseline_model'](a,device)
        if c['variant']=='reference_response':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(c['model_seed']);m.id_backbone=Phase1IdentityAdapter(a).to(device)
            if m.emb_dim!=160:raise ValueError('Identity dimension mismatch')
        # All controls retain the native nuisance heads; disabled losses give
        # them no task gradient. This keeps mechanism comparisons matched.
        return m
    def pairs(l,u,*,use_muse,use_unlabeled_step_budget=False):
        return originals['_muse_epoch_pairs'](l,u,use_muse=use_muse,use_unlabeled_step_budget=True)
    n.build_baseline_model=build;n.WiSigSubsetDataset=HiddenU
    n._resolve_unlabeled_batch_size=lambda a:256
    n._unlabeled_drop_last=lambda a:False
    n._muse_epoch_pairs=pairs
    n.resolve_device=lambda device:torch.device(device if torch.cuda.is_available() else 'cpu')
    try:yield n
    finally:
        for k,v in originals.items():setattr(n,k,v)
