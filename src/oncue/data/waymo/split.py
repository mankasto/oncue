"""Fixed split of WOMD scenes into train and internal-validation partitions.

The split hashes the scene id, so a scene lands in the same partition on every machine.
"""

from __future__ import annotations

import hashlib

VALIDATION_FRACTION = 0.2
HASH_HEX_DIGITS = 16  # first 64 bits of the hash, read as a fraction of 2**64


def partition(scene_id: str, validation_fraction: float = VALIDATION_FRACTION) -> str:
    """Return ``"internal_validation"`` or ``"train"`` for ``scene_id``."""
    value = int(hashlib.sha256(scene_id.encode()).hexdigest()[:HASH_HEX_DIGITS], 16) / 2**64
    return "internal_validation" if value < validation_fraction else "train"
