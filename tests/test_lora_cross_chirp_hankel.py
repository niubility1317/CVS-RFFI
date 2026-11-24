import torch

from experiments.pamf_lora_main.cross_chirp_hankel import (
    EMBEDDING, LOCAL_HOP, LOCAL_WIDTH, apply_subspace, match_dose, subspace_filter,
)


def test_hankel_overlap_add_identity_exact_at_both_edges():
    rng=torch.Generator().manual_seed(95)
    wave=torch.randn(2,6,256,dtype=torch.complex64,generator=rng)
    eye=torch.eye(EMBEDDING,dtype=wave.dtype)
    torch.testing.assert_close(apply_subspace(wave,eye,LOCAL_WIDTH,LOCAL_HOP),wave)
    torch.testing.assert_close(apply_subspace(wave,eye,256,256),wave)


def test_excluded_chirp_does_not_supply_its_own_subspace():
    rng=torch.Generator().manual_seed(29)
    wave=torch.randn(1,6,256,dtype=torch.complex64,generator=rng)
    scale=torch.full((1,6,1),.35)
    old=subspace_filter(wave,scale,LOCAL_WIDTH,LOCAL_HOP,exclude_self=True)
    altered=wave.clone()
    altered[:,2]*=1.3
    updated=subspace_filter(altered,scale,LOCAL_WIDTH,LOCAL_HOP,exclude_self=True)
    torch.testing.assert_close(updated[:,2],old[:,2],atol=1e-6,rtol=1e-5)
    assert (updated[:,0]-old[:,0]).abs().max()>.001


def test_freq_direction_survives_and_orthogonal_probe_shrinks_for_repeated_tone():
    t=torch.arange(256,dtype=torch.float64)
    good=torch.exp(2j*torch.pi*t/12)
    bad=torch.exp(-2j*torch.pi*t/8)
    tone=good[None,None].expand(1,6,-1)
    std=torch.ones(1,6,1,dtype=torch.float64)
    filt=subspace_filter(tone,std,LOCAL_WIDTH,LOCAL_HOP,exclude_self=True)
    keep=apply_subspace(tone,filt,LOCAL_WIDTH,LOCAL_HOP)
    suppress=apply_subspace(bad[None,None].expand_as(tone),filt,LOCAL_WIDTH,LOCAL_HOP)
    torch.testing.assert_close(keep,tone*(1-1/EMBEDDING))
    torch.testing.assert_close(suppress,torch.zeros_like(tone),atol=1e-12,rtol=0)


def test_same_dose_is_capped_and_matches_per_chirp():
    base=torch.tensor([[1j,2+0j],[.5j,1j],[0j,0j]])
    edited=torch.tensor([[.5j,0j],[2j,3j],[1+1j,3+0j]])
    protected, control=match_dose(base,edited)
    norm=lambda x:torch.linalg.vector_norm(x,dim=-1)
    assert (norm(protected)<=norm(base)+1e-6).all()
    torch.testing.assert_close(norm(protected),norm(control))
