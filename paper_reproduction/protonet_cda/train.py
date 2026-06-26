from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path

import torch
from torch import nn

from paper_reproduction.common.config import contains_unresolved_placeholder, contains_unspecified, load_json_config
from paper_reproduction.common.wisig_runtime import (
    collate_wisig,
    load_wisig_compact_pkl,
    make_wisig_trainval_test_by_day_rx,
    sample_to_dict,
    set_seed,
    write_json,
)
from paper_reproduction.protonet_cda.model import compute_prototypes, distance_logits, prototypical_nll


class ProtoEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _group_indices(dataset) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for i in range(len(dataset)):
        y = int(sample_to_dict(dataset[i])["label"])
        groups.setdefault(y, []).append(i)
    return groups


def _sample_episode(dataset, groups, *, n_way: int, k_shot: int, query_per_class: int, rng: random.Random, device):
    labels = [label for label, idxs in groups.items() if len(idxs) >= k_shot + query_per_class]
    if len(labels) < n_way:
        raise ValueError(f"not enough classes for {n_way}-way {k_shot}-shot episode")
    chosen = rng.sample(labels, n_way)
    support = []
    query = []
    for label in chosen:
        idxs = rng.sample(groups[label], k_shot + query_per_class)
        support.extend(dataset[i] for i in idxs[:k_shot])
        query.extend(dataset[i] for i in idxs[k_shot:])
    sb = collate_wisig(support)
    qb = collate_wisig(query)
    return sb["iq"].to(device), sb["label"].to(device), qb["iq"].to(device), qb["label"].to(device)


def _evaluate_target(model, dataset, groups, *, n_way: int, k_shot: int, query_per_class: int, episodes: int, seed: int, device) -> float:
    rng = random.Random(seed)
    accs = []
    model.eval()
    with torch.no_grad():
        for _ in range(max(1, episodes)):
            sx, sy, qx, qy = _sample_episode(dataset, groups, n_way=n_way, k_shot=k_shot, query_per_class=query_per_class, rng=rng, device=device)
            _, pred = prototypical_nll(model(sx), sy, model(qx), qy, metric="euclidean")
            accs.append(float((pred == qy).float().mean().item()))
    return float(sum(accs) / max(1, len(accs)))


def _make_split_compat(ds, **kwargs):
    accepted = inspect.signature(make_wisig_trainval_test_by_day_rx).parameters
    return make_wisig_trainval_test_by_day_rx(
        ds,
        **{k: v for k, v in kwargs.items() if k in accepted},
    )


def _run(config: dict, args: argparse.Namespace) -> dict:
    seed = int(args.seed if args.seed is not None else config.get("seed", 1337))
    set_seed(seed)
    rng = random.Random(seed)
    device = torch.device(args.device)
    wisig_pkl = str(args.wisig_pkl or config.get("wisig_pkl", ""))
    if not wisig_pkl:
        raise ValueError("wisig_pkl is required for ProtoNet training")
    ds = load_wisig_compact_pkl(wisig_pkl)
    train_ds, val_ds, test_ds, named_tests, named_meta, split_info = _make_split_compat(
        ds,
        equalized=int(config.get("equalized", 1)),
        out_len=256,
        domain="day",
        train_ratio=float(config.get("train_ratio", 0.1)),
        train_days=config.get("source_days"),
        test_days=config.get("target_days"),
        train_rxs=config.get("source_receivers"),
        test_rxs=config.get("target_receivers"),
        max_samples_per_combo_train=int(config.get("max_samples_per_combo_train", 40)),
        max_samples_per_combo_val=int(config.get("max_samples_per_combo_val", 20)),
        max_samples_per_combo_test=int(config.get("max_samples_per_combo_test", 40)),
        seed=seed,
        split_strategy="random",
        cap_strategy="random",
    )
    n_way = int(config.get("n_way", 6))
    k_shot = int(config.get("k_shot", 5))
    query_per_class = int(config.get("query_per_class", 5))
    model = ProtoEmbeddingNet(embedding_dim=int(config.get("embedding_dim", 128))).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=float(config.get("learning_rate", 0.01)))
    train_groups = _group_indices(train_ds)
    target_ds = named_tests.get("test_unseen_day_unseen_rx", test_ds)
    target_groups = _group_indices(target_ds)
    epochs = int(args.epochs or config.get("epochs", 1))
    max_steps = int(args.max_steps or config.get("max_steps", 0))
    global_step = 0
    last_loss = 0.0
    for _ in range(epochs):
        model.train()
        steps_this_epoch = max_steps if max_steps else int(config.get("steps_per_epoch", 50))
        for _ in range(steps_this_epoch):
            sx, sy, qx, qy = _sample_episode(
                train_ds,
                train_groups,
                n_way=n_way,
                k_shot=k_shot,
                query_per_class=query_per_class,
                rng=rng,
                device=device,
            )
            opt.zero_grad(set_to_none=True)
            loss, _ = prototypical_nll(model(sx), sy, model(qx), qy, metric="euclidean")
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().item())
            global_step += 1
            if max_steps and global_step >= max_steps:
                break
        if max_steps and global_step >= max_steps:
            break
    result = {
        "baseline": "protonet_cda",
        "seed": seed,
        "steps": global_step,
        "last_loss": last_loss,
        "target_episode_acc": _evaluate_target(
            model,
            target_ds,
            target_groups,
            n_way=n_way,
            k_shot=k_shot,
            query_per_class=query_per_class,
            episodes=int(config.get("eval_episodes", 5)),
            seed=seed + 17,
            device=device,
        ),
        "split_info": split_info,
        "named_test_meta": named_meta,
    }
    out_dir = Path(args.run_dir)
    write_json(out_dir / "metrics.json", result)
    write_json(out_dir / "resolved_config.json", config)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-original ProtoNet CDA WiSig entrypoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--wisig-pkl", default="")
    parser.add_argument("--run-dir", default="runs/paper_reproduction_protonet")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_json_config(args.config)
    if args.formal and contains_unspecified(config):
        raise ValueError("formal ProtoNet config still contains paper-unspecified")
    if args.formal and contains_unresolved_placeholder(config):
        raise ValueError("formal ProtoNet config still contains unresolved placeholder")
    if args.dry_run:
        print(json.dumps({"baseline": "protonet_cda", "config": config}, ensure_ascii=False, sort_keys=True))
        return 0
    result = _run(config, args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
