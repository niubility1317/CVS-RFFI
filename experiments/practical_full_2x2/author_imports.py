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


class _RaggedRandom:
    def __getattr__(self,name):
        import numpy as np
        return getattr(np.random,name)

    def permutation(self,value):
        import numpy as np
        if isinstance(value,(list,tuple)) and value and any(np.shape(v)!=np.shape(value[0]) for v in value):
            objects=np.empty(len(value),dtype=object);objects[:]=value;value=objects
        return np.random.permutation(value)


class _RaggedNumpy:
    random=_RaggedRandom()
    def __getattr__(self,name):
        import numpy as np
        return getattr(np,name)


def compatible_sdg_author(author):
    """Keep the author's function/RNG; restore legacy ragged object coercion."""
    from types import FunctionType
    original=next(op for op in author.augmentations if op.__name__=='permutation')
    if getattr(original,'_practical_sdg_array_compat',False):return
    fixed=FunctionType(original.__code__,dict(original.__globals__,np=_RaggedNumpy()),
        original.__name__,original.__defaults__,original.__closure__)
    fixed._practical_sdg_array_compat=True
    author.augmentations=[fixed if op is original else op for op in author.augmentations]
