"""Run independent jobs in this process or in a process pool."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from multiprocessing.context import BaseContext


def run_jobs(
    job: Callable[[Any], Any],
    items: Iterable[Any],
    workers: int,
    *,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
    context: BaseContext | None = None,
    ordered: bool = False,
) -> Iterator[Any]:
    """Yield `job(item)` for every item, here with one worker or else in a process pool.

    Pool results come as they finish unless `ordered`. `initializer(*initargs)` runs once per
    process (here too); `context` is the start method (None: the platform's default).
    """
    if workers <= 1:
        if initializer is not None:
            initializer(*initargs)
        yield from map(job, items)
        return
    with ProcessPoolExecutor(
        workers, mp_context=context, initializer=initializer, initargs=initargs
    ) as pool:
        if ordered:
            yield from pool.map(job, items)
        else:
            for future in as_completed([pool.submit(job, item) for item in items]):
                yield future.result()
