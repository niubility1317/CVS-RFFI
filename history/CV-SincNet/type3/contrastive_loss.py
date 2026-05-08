# contrastive_loss.py
import torch
import torch.nn.functional as F


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Supervised Contrastive Loss (SupCon), cosine similarity.
    features: (B, V, D)
    labels:   (B,)
    """
    device = features.device
    B, V, D = features.shape

    # (B*V, D)
    feats = F.normalize(features.reshape(B * V, D), dim=1)

    # cosine sim / tau
    logits = feats @ feats.T
    logits = logits / temperature

    # stability
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # mask positives: same class
    labels = labels.view(B, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  # (B,B)

    # expand to views: (B*V,B*V)
    mask = mask.repeat_interleave(V, dim=0).repeat_interleave(V, dim=1)

    # remove self
    self_mask = torch.eye(B * V, device=device)
    mask = mask * (1.0 - self_mask)

    exp_logits = torch.exp(logits) * (1.0 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_cnt = mask.sum(dim=1)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos_cnt + 1e-12)
    return (-mean_log_prob_pos).mean()


class PrototypeMemory(torch.nn.Module):
    """
    EMA class prototypes: protos[c] = normalized center for class c.
    Use detached features to update.
    """
    def __init__(self, num_classes: int, feat_dim: int, momentum: float = 0.9):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.momentum = float(momentum)

        self.register_buffer("protos", torch.zeros(self.num_classes, self.feat_dim))
        self.register_buffer("counts", torch.zeros(self.num_classes, dtype=torch.long))

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor):
        """
        feats: (B, D) - detached recommended
        labels: (B,)
        """
        feats = F.normalize(feats, dim=1)
        labels = labels.view(-1)

        for c in labels.unique():
            c = int(c.item())
            idx = (labels == c)
            if idx.sum() == 0:
                continue
            batch_center = feats[idx].mean(dim=0)
            batch_center = F.normalize(batch_center, dim=0)

            if self.counts[c] == 0:
                self.protos[c] = batch_center
            else:
                self.protos[c] = F.normalize(
                    self.momentum * self.protos[c] + (1.0 - self.momentum) * batch_center, dim=0
                )
            self.counts[c] += idx.sum()

    def get(self):
        return self.protos, self.counts


def prototype_contrastive_loss(
    feats: torch.Tensor,
    labels: torch.Tensor,
    protos: torch.Tensor,
    counts: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Prototype contrast (cosine InfoNCE on class centers).
    feats:   (N, D)  (can be B*V)
    labels:  (N,)
    protos:  (C, D)
    counts:  (C,) to mask uninitialized prototypes
    """
    device = feats.device
    feats = F.normalize(feats, dim=1)
    protos = F.normalize(protos.to(device), dim=1)

    logits = (feats @ protos.T) / temperature  # (N,C)

    # mask classes not initialized
    valid = (counts > 0).to(device=device)
    if valid.sum() < 2:
        # too few prototypes to contrast
        return torch.zeros([], device=device)

    # set invalid columns to very negative
    invalid_cols = ~valid
    logits[:, invalid_cols] = -1e9

    loss = F.cross_entropy(logits, labels)
    return loss
