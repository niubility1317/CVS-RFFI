"""Minimal namespace bootstrap for an isolated D127 runner source root.

The runner may stage D127 modules beside a separate root that supplies the
legacy D106 modules.  Extending the package path keeps both roots importable
without importing project modules or touching scientific state.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
