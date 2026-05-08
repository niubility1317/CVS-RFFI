import torch

from model_dual_cvsincnet import DualCVSincNetDisentangle
from sgc_adapter import (
    AmplitudeNormalizer,
    FPCRSGCReconstructor,
    FrequencyOffsetCompensator,
    FingerprintPreservingChannelProjector,
    ResidualChannelCompensator,
    SGCAdapter,
    SpectralInterferenceSuppressor,
)
from sgc_losses import (
    SourcePrototypeBank,
    feature_consistency_loss,
    fpcr_budget_regularization,
    fpcr_physics_preservation_loss,
    pseudo_label_loss,
    residual_regularization,
)


def test_sgc_adapter_preserves_shape_and_gradients():
    adapter = SGCAdapter()
    x = torch.randn(4, 2, 256, requires_grad=True)

    out, aux = adapter(x, return_aux=True)
    out.square().mean().backward()

    assert out.shape == x.shape
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert "adapter_output" in aux


def test_sgc_adapter_submodule_toggles_preserve_shape():
    x = torch.randn(2, 2, 128)
    for amp in (True, False):
        for freq in (True, False):
            for spectral in (True, False):
                for residual in (True, False):
                    adapter = SGCAdapter(
                        use_amp_norm=amp,
                        use_freq_comp=freq,
                        use_spectral_suppressor=spectral,
                        use_residual_comp=residual,
                    )
                    out, _ = adapter(x)
                    assert out.shape == x.shape


def test_individual_sgc_blocks_shape_and_parameter_budget():
    x = torch.randn(3, 2, 256)

    normed = AmplitudeNormalizer()(x * 10.0)
    rms = torch.sqrt(torch.mean(normed.square(), dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    assert FrequencyOffsetCompensator()(x).shape == x.shape
    assert SpectralInterferenceSuppressor()(x).shape == x.shape
    assert ResidualChannelCompensator()(x).shape == x.shape

    n_params = sum(p.numel() for p in SGCAdapter().parameters())
    assert n_params < 50_000


def test_residual_regularization_and_losses_are_safe_when_masks_empty():
    adapter = SGCAdapter(use_amp_norm=False, use_freq_comp=False, use_spectral_suppressor=False)
    _, aux = adapter(torch.randn(2, 2, 64), return_aux=True)
    assert residual_regularization(aux).ndim == 0

    clean = torch.randn(4, 8)
    shifted = clean + 0.1
    assert feature_consistency_loss(clean, shifted).ndim == 0

    logits = torch.zeros(4, 5)
    loss, mask, conf_mean, ratio = pseudo_label_loss(logits, threshold=0.99)
    assert loss.ndim == 0
    assert not mask.any()
    assert conf_mean > 0.0
    assert ratio == 0.0


def test_source_prototype_bank_updates_and_aligns():
    bank = SourcePrototypeBank(num_classes=3, feat_dim=4)
    z = torch.randn(6, 4)
    y = torch.tensor([0, 1, 1, 2, 2, 2])

    bank.update(z, y)

    assert bank.initialized.all()
    assert bank.get(y).shape == z.shape
    assert bank.alignment_loss(z, y).ndim == 0


def test_fpcr_channel_projector_preserves_shape_and_reports_physics_stats():
    projector = FingerprintPreservingChannelProjector(
        shrinkage=0.35,
        cepstral_lifter=8,
        occupied_band_fraction=0.70,
    )
    x = torch.randn(3, 2, 128)

    out, aux = projector(x, return_aux=True)

    assert out.shape == x.shape
    assert aux["fpcr_projection_delta_rms"].ndim == 0
    assert aux["fpcr_spectral_regrowth_ratio_in"].ndim == 0
    assert aux["fpcr_spectral_regrowth_ratio_out"].ndim == 0
    assert aux["fpcr_iq_image_ratio_in"].ndim == 0
    assert aux["fpcr_cepstral_detail_energy_out"].ndim == 0


def test_fpcr_reconstructor_bounds_learned_residual_ratio():
    recon = FPCRSGCReconstructor(
        shrinkage=0.20,
        cepstral_lifter=6,
        residual_channels=12,
        residual_blocks=1,
        max_residual_ratio=0.06,
        residual_init_gamma=1.0,
    )
    x = torch.randn(4, 2, 96, requires_grad=True)

    out, aux = recon(x, return_aux=True)
    out.square().mean().backward()

    assert out.shape == x.shape
    assert x.grad is not None
    assert aux["fpcr_residual_ratio"].item() <= 0.061
    assert aux["fpcr_budget_loss"].ndim == 0


def test_sgc_adapter_can_use_fpcr_mode_and_losses():
    adapter = SGCAdapter(
        adapter_mode="fpcr",
        fpcr_shrinkage=0.25,
        fpcr_cepstral_lifter=6,
        fpcr_max_residual_ratio=0.06,
    )
    x = torch.randn(2, 2, 128)

    out, aux = adapter(x, return_aux=True)
    loss_phys = fpcr_physics_preservation_loss(aux)
    loss_budget = fpcr_budget_regularization(aux)

    assert out.shape == x.shape
    assert "fpcr_projected" in aux
    assert loss_phys.ndim == 0
    assert loss_budget.ndim == 0


def test_dual_model_exposes_sgc_aux_when_enabled():
    model = DualCVSincNetDisentangle(
        num_classes=4,
        num_domains=2,
        model_size="S",
        input_len=128,
        model_variant="lite_b",
        branch_ablation="no_dac",
        sgc_adapter=True,
        sgc_adapter_kwargs={
            "use_amp_norm": False,
            "use_freq_comp": False,
            "use_spectral_suppressor": False,
            "use_residual_comp": True,
        },
    )
    x = torch.randn(2, 2, 128)
    y = torch.tensor([0, 1])

    out = model(x, y_tx=y, return_aux=True, domain_labels=torch.tensor([0, 1]))

    assert out["tx_logits"].shape == (2, 4)
    assert out["sgc_aux"]["adapter_output"].shape == x.shape
