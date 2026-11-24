"""Make and label hidden-target tests: `generate` builds them on scene shards, `label.label`
runs the oracle on each sample."""

from .label import usable
from .pipeline import generate
from .targets import SPECS

__all__ = ["SPECS", "generate", "usable"]
