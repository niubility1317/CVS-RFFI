"""Build the paper-aligned cache as verified archives arrive, then train."""

import argparse
from pathlib import Path

from .paper_cache import build_cache
from .paper_train import launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    args = parser.parse_args()
    data, out = args.data.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    source = data / "manifest" / "source"
    cache = data / "cache_source"
    build_cache("source", data / "archives", source, cache, workers=4,
                heldout_manifest=data / "manifest" / "heldout_iq", heldout_out=data / "cache_test",
                wait_archive_seconds=5400)
    launch(args.config.resolve(), source, cache, out, args.gpus)
