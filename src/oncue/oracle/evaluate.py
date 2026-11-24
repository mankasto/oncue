"""The budgeted oracle (``Oracle.judge``): does some candidate avoid the target?

A condition is found when some candidate is safe. Road users replay the scene and do not react
to the ego.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import shapely

from ..core.config import OracleConfig
from ..core.types import Scene
from ..core.vehicle import execute, initial_steering
from ..data.map_geometry import drivable_region, swept_area
from ..utils.geometry import union, union_all
from ..utils.tracks import substep_states
from . import checks
from .responses import Candidate, build, candidate_list, list_digest, prefix

if TYPE_CHECKING:  # for annotations only: no new import when running
    from ..core.types import Track
    from ..core.vehicle import Execution
    from .responses import Built

ROAD_TOLERANCE_M = 0.25  # how far a footprint may poke out of the drivable region
CONTENT_DIGEST_BYTES = 12  # byte length of a track's content digest used as a cache key


class Oracle:
    """Judges information conditions over a scene, caching the last scene's road geometry."""

    ROAD_TYPES = ("vehicle",)  # road users whose driven area counts as evidence of road

    def __init__(self, config: OracleConfig):
        self.config = config
        self.candidates = candidate_list(config.candidates)
        self.core_size = sum(c.core for c in self.candidates)
        self.digest = list_digest(self.candidates)
        self._cached_scene: str | None = None
        self._maps: dict[str, object] = {}  # approximate map region per scene
        self._swept: dict[tuple[str, bytes], object] = {}  # swept area per track id and content
        self._regions: dict[tuple, object] = {}
        self._roads: dict[str, checks.Roads] = {}
        self._profiles: dict = {}  # longitudinal profiles keyed by their inputs

    def judge(
        self,
        scene: Scene,
        target_id: str,
        start: int,
        end: int,
        budget: int | None = None,
        done: Judgement | None = None,
    ) -> Judgement:
        """Judge the first ``budget`` candidates from frame ``start`` over ``[start, end)``.

        ``budget`` None means the config's. ``done``, a judgement of a shorter prefix over the
        same window, is extended rather than rerun, and its seconds are added in.
        """
        cfg, clock = self.config, time.perf_counter()
        if not 0 < start < end - 1 or end > len(scene.timestamps):
            raise ValueError(f"window [{start}, {end}) of {len(scene.timestamps)} frames")
        picked = prefix(self.candidates, cfg.budget if budget is None else budget)
        kept, earlier_s = (), 0.0
        if done is not None:
            kept, earlier_s = done.outcomes, done.elapsed_s
            same = [o.candidate for o in kept] == list(picked[: len(kept)])
            if (done.start, done.end) != (start, end) or not same:
                raise ValueError("`done` judged another window or another prefix of the list")
        before = self._before_start(scene, target_id, start)
        if before is not None:
            elapsed = time.perf_counter() - clock + earlier_s
            status = BEFORE_START[0]
            return Judgement(status, start, end, (), len(picked), elapsed, before_start=before)
        target = scene.tracks[target_id]
        rest = picked[len(kept) :]
        self._use(scene.scene_id)
        built = build(scene, start, end, rest, cfg, target, self._profiles)
        outcomes = [*kept, *(Outcome(c, "not_applicable") for c in rest)]
        times = done.times if done is not None else None
        if built.rows:
            run = execute(
                built.desired,
                scene.timestamps[start:end],
                initial_steering(scene.ego, start, scene.dt, cfg.vehicle, scene.anchor),
                cfg.vehicle,
                built.commands,
            )
            times = run.times
            tried = [rest[i] for i in built.rows]
            checked = self._check(scene, target_id, start, run, built, tried)
            for i, outcome in zip(built.rows, checked, strict=True):
                outcomes[len(kept) + i] = outcome
        codes = "".join(o.code for o in outcomes)
        return Judgement(
            status_of(codes),
            start,
            end,
            tuple(outcomes),
            len(picked),
            time.perf_counter() - clock + earlier_s,
            times,
        )

    def _check(
        self,
        scene: Scene,
        target_id: str,
        start: int,
        run: Execution,
        built: Built,
        tried: list[Candidate],
    ) -> list[Outcome]:
        """Return the outcome for every executed row; ``tried`` names the candidates in order."""
        cfg = self.config
        sample = checks.samples(run, cfg.vehicle.substeps, cfg.collision_substeps)
        sampled, times = run.states[:, sample], run.times[sample]
        violations = checks.dynamics_violations(run, cfg)
        outside = checks.outside_region(sampled, scene.ego.size, self._region(scene, target_id))
        roads = self._roads_of(scene)
        found = checks.clearances(sampled, scene.ego.size, times, scene, roads, (target_id,))
        rows = len(sampled)
        background, nearest = found.nearest(rows, skip=target_id)
        target = found.per_actor.get(target_id, np.full(rows, checks.CLEARANCE_CAP_M))
        every_half_second = max(1, round(0.5 * cfg.vehicle.substeps / scene.dt))
        opposing = roads.lanes.opposing(run.states[:, ::every_half_second])
        speeding = roads.speeding(run.states[..., 3], cfg.speeding_tol_mps)
        out = []
        for r in range(rows):
            why = violations[r] + (("offroad",) if outside[r] else ())
            if why:
                result = "infeasible"
            elif background[r] <= 0:
                result = "background"
            elif target[r] <= 0:
                result = "target"
            else:
                result = "safe"
            out.append(
                Outcome(
                    candidate=tried[r],
                    result=result,
                    violations=why,
                    executed=run.states[r],
                    target_clearance_m=float(target[r]),
                    background_clearance_m=float(background[r]),
                    nearest_id=nearest[r],
                    opposing=bool(opposing[r]),
                    speeding=bool(speeding[r]),
                    excused_by=tuple(found.excused_by(r)),
                    stop_accel=built.stop_accel[r],
                )
            )
        return out

    def _region(self, scene: Scene, target_id: str):
        """Return the map area plus the swept area of every vehicle but the target.

        The target's track may be generated, so it never counts as evidence of road.
        """
        self._use(scene.scene_id)
        if scene.scene_id not in self._maps:
            self._maps[scene.scene_id] = drivable_region(scene)
        tracks = [t for t in scene.others(target_id) if t.agent_type in self.ROAD_TYPES] + (
            [scene.ego] if scene.ego.agent_type in self.ROAD_TYPES else []
        )
        keys = [(t.track_id, _content(t)) for t in tracks]
        key = tuple(sorted(keys))
        if key not in self._regions:
            for k, t in zip(keys, tracks, strict=True):
                if k not in self._swept:
                    self._swept[k] = swept_area(t)
            driven = union_all([self._swept[k] for k in keys])
            region = union(self._maps[scene.scene_id], driven)
            region = region.buffer(ROAD_TOLERANCE_M)  # absorb map and footprint approximations
            shapely.prepare(region)
            self._regions[key] = region
        return self._regions[key]

    def _use(self, scene_id: str) -> None:
        """Switch to ``scene_id`` and drop any cached geometry or profiles from another scene."""
        if scene_id != self._cached_scene:
            for cache in (self._maps, self._swept, self._regions, self._roads, self._profiles):
                cache.clear()
            self._cached_scene = scene_id

    def _roads_of(self, scene: Scene) -> checks.Roads:
        """Return the scene's cached ``checks.Roads`` (lanes and road edges)."""
        self._use(scene.scene_id)
        if scene.scene_id not in self._roads:
            self._roads[scene.scene_id] = checks.Roads.of(scene)
        return self._roads[scene.scene_id]

    def _before_start(self, scene: Scene, target_id: str, start: int):
        """Check the reference ego from the anchor up to ``start`` for any contact.

        Returns ``(road_user, frame)`` of the earliest, or None.
        """
        cfg, anchor = self.config, scene.anchor
        if start <= anchor:
            return None
        per = cfg.collision_substeps
        times, ego, valid = substep_states(scene.ego, scene.timestamps, anchor, start, per)
        kept = np.flatnonzero(valid)  # valid samples' positions among all samples from the anchor
        states = ego[kept][None]
        found = checks.clearances(states, scene.ego.size, times[kept], scene, None, (target_id,))
        # Earliest contact first (ties by road user id), at the end of the frame step it is in.
        hits = sorted(
            (anchor + int(np.ceil(kept[first[0]] / per)), track_id)
            for track_id, first in found.first_contact.items()
            if first[0] >= 0
        )
        return (hits[0][1], hits[0][0]) if hits else None


def _content(track: Track) -> bytes:
    """Return a short content digest of a track's motion for keying derived geometry."""
    h = hashlib.blake2b(digest_size=CONTENT_DIGEST_BYTES)
    h.update(np.ascontiguousarray(track.states).tobytes())
    h.update(np.ascontiguousarray(track.valid).tobytes())
    return h.digest()


# A candidate's result as one letter. Everything a condition reports is read from these codes,
# so a stored run cut to its first k codes gives the results of that smaller budget.
CODES = {"safe": "S", "target": "T", "background": "B", "infeasible": "I", "not_applicable": "N"}
BACKGROUND_SAFE = "ST"  # feasible and clear of every other road user
BEFORE_START = ("collision_before_start",)  # judgement halted before any candidate was tried


@dataclass(frozen=True, eq=False)
class Judgement:
    """One information-condition judgement: window, candidates tried, outcomes."""

    status: str  # ``found`` | ``not_found`` | one of ``BEFORE_START``
    start: int
    end: int  # exclusive
    outcomes: tuple[Outcome, ...]
    tried: int  # list entries considered (whole packs); 0 when halted before the start
    elapsed_s: float
    times: np.ndarray | None = None  # dense execution times
    before_start: tuple | None = None  # (road_user, contact frame) when halted early

    @property
    def codes(self) -> str:
        """Return the outcomes' codes as a single string."""
        return "".join(o.code for o in self.outcomes)

    def witness(self) -> Outcome | None:
        """Return the first safe candidate, preferring lawful ones; None when none is safe."""
        safe = [o for o in self.outcomes if o.result == "safe"]
        return next((o for o in safe if o.lawful), safe[0] if safe else None)

    def unlawful_only(self) -> bool:
        """Return whether every safe candidate drives against traffic or speeds."""
        safe = [o for o in self.outcomes if o.result == "safe"]
        return bool(safe) and not any(o.lawful for o in safe)


@dataclass(frozen=True, eq=False)
class Outcome:
    """Result for one candidate within a ``Judgement``."""

    candidate: Candidate
    result: str  # a key of ``CODES``
    violations: tuple[str, ...] = ()  # subset of {``tracking``, ``speed``, ``offroad``}
    executed: np.ndarray | None = None  # ``[M, 5]`` dense executed states
    target_clearance_m: float = float("nan")
    background_clearance_m: float = float("nan")  # to anyone but the target; capped
    nearest_id: str | None = None
    opposing: bool = False  # entered an opposing lane
    speeding: bool = False  # exceeded the posted limit
    excused_by: tuple[str, ...] = ()  # road users whose contact was not the ego's fault
    stop_accel: float | None = None  # deceleration solved for a stop at the conflict, if any

    @property
    def code(self) -> str:
        """Return the single-letter code for this outcome."""
        return CODES[self.result]

    @property
    def lawful(self) -> bool:
        """Return whether this outcome neither drove against traffic nor exceeded the limit."""
        return not (self.opposing or self.speeding)


def status_of(codes: str) -> str:
    """Return ``"found"`` when some candidate is safe, otherwise ``"not_found"``."""
    return "found" if "S" in codes else "not_found"


def counts(codes: str) -> dict:
    """Return the per-status counts summarising a ``codes`` string."""
    first = codes.find("S")
    return {
        "attempted": len(codes),
        "not_applicable": codes.count("N"),
        "executed": len(codes) - codes.count("N"),
        "feasible": sum(c not in "NI" for c in codes),
        "background_safe": sum(c in BACKGROUND_SAFE for c in codes),
        "safe": codes.count("S"),
        "first_safe_rank": None if first < 0 else first,
    }


def background_safe(codes: str) -> bool:
    """Return whether any candidate was feasible and clear of every other road user."""
    return any(c in BACKGROUND_SAFE for c in codes)
