"""Shared analysis code: reading outputs, run outcomes, rates and intervals.

``output_rows`` reads output files, ``run_flags`` says what a closed-loop run counts as, and
``rate`` gives a share with its interval (call ``use_drives`` first to resample by drive).
"""

from .bootstrap import Estimate, use_drives
from .outputs import output_rows
from .rates import rate, run_flags

__all__ = ["Estimate", "output_rows", "rate", "run_flags", "use_drives"]
