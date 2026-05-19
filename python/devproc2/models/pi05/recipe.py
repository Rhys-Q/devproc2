"""Compatibility re-export for Pi0.5 model export declarations."""
from __future__ import annotations

from devproc2.models.pi05 import export_spec as _export_spec
from devproc2.models.pi05.export_spec import *  # noqa: F403

__all__ = _export_spec.__all__
