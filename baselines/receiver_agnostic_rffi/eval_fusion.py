from __future__ import annotations

import argparse

import torch

from baselines.receiver_agnostic_rffi.collaborative_inference import adaptive_soft_fusion, soft_fusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Receiver collaborative fusion smoke utility")
    parser.add_argument("--method", choices=["soft", "adaptive_soft"], default="soft")
    args = parser.parse_args()
    probs = torch.softmax(torch.randn(3, 4), dim=1)
    if args.method == "soft":
        fused = soft_fusion(probs)
    else:
        fused = adaptive_soft_fusion(probs, torch.tensor([10.0, 20.0, 30.0]))
    print(fused.tolist())


if __name__ == "__main__":
    main()
