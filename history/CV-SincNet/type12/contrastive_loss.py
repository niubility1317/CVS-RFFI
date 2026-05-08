# contrastive_loss.py
import torch
import torch.nn.functional as F

def supcon_loss(features: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    Stable SupCon (compute in FP32).
    features: (B, V, D)
    labels:   (B,)
    """
    if features.dim() != 3:
        raise ValueError(f"features must be (B,V,D), got {features.shape}")

    device = features.device
    B, V, D = features.shape

    # >>> critical: always compute in fp32
    features = features.float()
    labels = labels.view(-1).long()

    # normalize for stable dot products
    features = F.normalize(features, dim=-1)

    # flatten views
    feats = features.reshape(B * V, D)  # (BV, D)
    labels_rep = labels.repeat_interleave(V)  # (BV,)

    # similarity
    logits = (feats @ feats.t()) / float(tau)  # (BV, BV)
    logits = logits - logits.max(dim=1, keepdim=True).values  # stability

    # masks
    self_mask = torch.eye(B * V, device=device, dtype=torch.bool)
    logits_mask = (~self_mask).float()

    pos_mask = (labels_rep.unsqueeze(0) == labels_rep.unsqueeze(1)).float()
    pos_mask = pos_mask * logits_mask  # remove self

    # log_prob
    exp_logits = torch.exp(logits) * logits_mask
    denom = exp_logits.sum(dim=1, keepdim=True)

    # eps must be meaningful in fp32 (1e-12 ok here)
    log_prob = logits - torch.log(denom + 1e-12)

    # mean over positives (avoid div0)
    pos_cnt = pos_mask.sum(dim=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_cnt + 1e-12)

    loss = -mean_log_prob_pos
    return loss.mean()


def prototype_contrastive_loss(
    feats: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    tau: float = 0.07,
    invalid_logit: float = -1e4,  # safe in fp16/fp32 (won't overflow to inf in fp16)
) -> torch.Tensor:
    """
    feats: (B, D)
    labels: (B,)
    prototypes: (C, D)  (may contain uninitialized rows)
    """
    feats = feats.float()
    prototypes = prototypes.float()
    labels = labels.view(-1).long()

    feats = F.normalize(feats, dim=1)
    prototypes = F.normalize(prototypes, dim=1)

    # detect uninitialized prototypes (all-zero or near-zero)
    invalid_cls = (prototypes.abs().sum(dim=1) < 1e-8)  # (C,)
    # if a sample's gt class is invalid, skip it (otherwise CE 会被迫学一个 -inf 类，极不稳定)
    valid_samples = ~invalid_cls[labels]

    if valid_samples.sum() == 0:
        return feats.new_tensor(0.0)

    feats_v = feats[valid_samples]
    labels_v = labels[valid_samples]

    logits = (feats_v @ prototypes.t()) / float(tau)  # (Bv, C)
    # mask invalid classes (safe constant)
    logits[:, invalid_cls] = float(invalid_logit)

    return F.cross_entropy(logits, labels_v)
