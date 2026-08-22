"""Internal CVS-RFFI training utilities.

The public script/module entrypoints stay at the repository root; this package
holds reusable implementation pieces shared by training, post-stage, and SGC/SSDG
entrypoints.
"""

from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)
