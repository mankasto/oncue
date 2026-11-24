"""Record the raw candidates of a subset of requests, each with its measured window W, to a file
next to the generation output (`raw_path`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .validity import (
    contact_frame,
    inserted,
    is_finite,
    reveal_check,
    tolerance_frames,
    window_error,
)

if TYPE_CHECKING:  # import only for type hints
    import numpy as np

    from ..context import SceneContext
    from ..motion.generators import Candidate
    from ..requests.build import Request, Start
    from ..targets import TargetSpec

# W_status of a measured window; otherwise the reason it was not measured
MEASURED = "measured"
NONFINITE = "nonfinite"  # the candidate's states are not finite
NOT_MEASURED = ("no_contact", "no_reveal", "glimpsed_before", "reoccluded")  # in table order
UNCOMPUTABLE = (NONFINITE, *NOT_MEASURED)  # every W_status but MEASURED


@dataclass
class RawCandidates:
    """One scene's diagnostic rows, one per subset request and learned generator."""

    every: int
    rows: list = field(default_factory=list)

    def __post_init__(self):
        if not (isinstance(self.every, int) and self.every >= 1):
            raise ValueError(f"raw candidates: one request in {self.every!r}, a whole number >= 1")

    def includes(self, request_id: str) -> bool:
        """Return whether `request_id` (hex) falls in the subset."""
        return int(request_id, 16) % self.every == 0

    def record(
        self,
        ctx: SceneContext,
        req: Request,
        spec: TargetSpec,
        request_id: str,
        generator: str,
        starts: list[Start],
        first: list[Candidate] | None,
    ) -> None:
        """Append a row with the first start and its candidates `first`, if `req` is in the
        subset."""
        if not self.includes(request_id):
            return
        row = {
            "request_id": request_id,
            "generator": generator,
            "W_req": req.W,
            "start": None,
            "candidates": [],
        }
        if starts:
            start = starts[0]
            row["start"] = {
                "profile": start.profile,
                "s": round(float(start.s), 3),
                "speed": round(float(start.speed), 3),
                "f_go": None if start.f_go is None else int(start.f_go),
                "s_reveal": None if start.s_reveal is None else round(float(start.s_reveal), 3),
            }
            row["candidates"] = [
                {
                    "mode": cand.mode,
                    "score": None if cand.score is None else round(cand.score, 3),
                    **measure_window(ctx, req, spec, cand.states, cand.valid),
                }
                for cand in first
            ]
        self.rows.append(row)


def measure_window(
    ctx: SceneContext, req: Request, spec: TargetSpec, states: np.ndarray, valid: np.ndarray
) -> dict:
    """Measure a candidate's reveal-to-contact window as `check_candidate` would.

    W_status is "measured" or why not; W_error is the measured minus the requested W, in seconds.
    """
    out = {
        "W_status": NONFINITE,
        "W": None,
        "W_error": None,
        "window_hit": None,
        "f_reveal": None,
        "f_contact": None,
    }
    if not is_finite(states, valid):
        return out
    f_c = contact_frame(ctx, spec, states, valid)
    if f_c is None:
        out["W_status"] = "no_contact"
        return out
    found = reveal_check(ctx, spec, inserted(ctx, spec, states, valid), f_c)
    reveal = None if found.event is None else int(found.event.reveal)
    out.update(W_status=found.reject or MEASURED, f_reveal=reveal, f_contact=f_c)
    if found.reject is None:
        off = window_error(req, f_c, reveal)
        out["W"] = round((f_c - reveal) * ctx.dt, 2)
        out["W_error"] = round(off * ctx.dt, 2)
        out["window_hit"] = abs(off) <= tolerance_frames(spec, ctx.dt)
    return out


def raw_path(out_path: str | Path) -> Path:
    """Return the diagnostic's companion path next to a generation output."""
    return Path(out_path).with_suffix(".raw_candidates.jsonl")
