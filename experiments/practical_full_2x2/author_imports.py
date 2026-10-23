"""Load the registered IB author's HSIC without replacing G's utils package."""
from functools import lru_cache
import importlib.util
from pathlib import Path


@lru_cache(maxsize=None)
def _hsic(path):
    spec=importlib.util.spec_from_file_location('_practical_ibrff_author_hsic',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.hsic_normalized_cca


def author_hsic(c):
    return _hsic(str(Path(c['ibrff_model_root']).resolve()/'utils/hsic.py'))
