"""Execution engine for `somnio.pipeline`.

## Multiprocessing strategy

When `parallel=True`, `execute()` can schedule independent steps concurrently.
Two backends are supported:

- **processes** (default): `concurrent.futures.ProcessPoolExecutor`
- **threads**: `concurrent.futures.ThreadPoolExecutor`

### Why processes by default

Most transforms are expected to be CPU-bound and NumPy-heavy. Processes provide
true parallelism (not limited by the GIL). The trade-off is **serialization
overhead**: inputs/outputs must be pickled to cross process boundaries, which
can be expensive for large arrays.

### Import-string transforms for robustness

To keep multiprocessing stable across platforms/start methods, the processes
backend requires each `Step.transforms` entry to be a `TransformSpec`
(import string + kwargs)
so workers only receive primitives + data bundles. The threads backend can run
direct callables (including closures), because it does not need pickling.

### When to choose threads

Use `backend="threads"` when transforms are I/O-bound, you want simpler
debugging, or a transform cannot be expressed as an import string.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from somnio.data import Epochs, Event, TimeSeries
from somnio.pipeline.types import (
    Bundle,
    OutputValue,
    Pipeline,
    Step,
    TransformSpec,
)
from somnio.pipeline.errors import (
    DeadEndError,
    OutputConflictError,
    PipelineExecutionError,
)
from somnio.utils.imports import resolve_import_string


Backend = Literal["processes", "threads"]


def _validate_out_bundle(
    step: Step, idx: int, out_bundle: Any
) -> dict[str, OutputValue]:
    if not isinstance(out_bundle, dict):
        raise PipelineExecutionError(
            f"Step [{idx}] {step.name} returned {type(out_bundle)!r}, expected dict[str, TimeSeries | Epochs | list[Event]]"
        )

    expected = set(step.outputs)
    got = set(out_bundle.keys())
    if got != expected:
        raise PipelineExecutionError(
            f"Step [{idx}] {step.name} returned keys {sorted(got)!r}, expected {sorted(expected)!r}"
        )

    for k, v in out_bundle.items():
        if not isinstance(k, str):
            raise PipelineExecutionError(
                f"Step [{idx}] {step.name} returned non-str key {k!r}"
            )

        if isinstance(v, (TimeSeries, Epochs)):
            continue

        if isinstance(v, list):
            bad = next((e for e in v if not isinstance(e, Event)), None)
            if bad is not None:
                raise PipelineExecutionError(
                    f"Step [{idx}] {step.name} output {k!r} contains {type(bad)!r}, expected Event"
                )
            continue

        raise PipelineExecutionError(
            f"Step [{idx}] {step.name} output {k!r} is {type(v)!r}, expected TimeSeries | Epochs | list[Event]"
        )

    return out_bundle


def _execute_transform(
    spec: TransformSpec,
    required_bundle: Bundle,
) -> Bundle:
    """Execute a transform on the required bundle.

    This is intentionally top-level and import-string-friendly so it can be used
    in thread/process worker contexts.
    """
    fn = (
        resolve_import_string(spec.target)
        if isinstance(spec.target, str)
        else spec.target
    )
    return fn(required_bundle, **spec.kwargs)


def _execute_step(
    *,
    idx: int,
    step: Step,
    required_bundle: Bundle,
) -> Bundle:
    current = required_bundle
    for t in step.transforms:
        current = _execute_transform(t, current)
    return _validate_out_bundle(step, idx, current)


@dataclass(frozen=True, slots=True)
class _Runnable:
    idx: int
    step: Step
    missing: tuple[str, ...]


def _missing_inputs(step: Step, available: set[str]) -> tuple[str, ...]:
    return tuple(name for name in step.inputs if name not in available)


def _iter_runnable(
    steps: Iterable[tuple[int, Step]], available: set[str]
) -> list[_Runnable]:
    runnable: list[_Runnable] = []
    for idx, step in steps:
        missing = _missing_inputs(step, available)
        if not missing:
            runnable.append(_Runnable(idx=idx, step=step, missing=()))
    return runnable


def _format_dead_end(
    remaining: list[tuple[int, Step]],
    available: set[str],
) -> str:
    parts: list[str] = []
    parts.append("No runnable steps; pipeline is stuck.")
    parts.append(f"Available keys: {sorted(available)!r}")
    parts.append("Missing inputs per remaining step:")
    for idx, step in remaining:
        missing = _missing_inputs(step, available)
        parts.append(f"- [{idx}] {step.name}: missing {sorted(missing)!r}")
    return "\n".join(parts)


def _detect_runnable_output_conflicts(runnable: list[_Runnable]) -> None:
    by_output: dict[str, list[_Runnable]] = {}
    for r in runnable:
        for out in r.step.outputs:
            by_output.setdefault(out, []).append(r)

    conflicts = {k: v for k, v in by_output.items() if len(v) > 1}
    if not conflicts:
        return

    lines: list[str] = ["Runnable output conflict detected:"]
    for out, rs in sorted(conflicts.items(), key=lambda kv: kv[0]):
        lines.append(
            f"- output {out!r} produced by: "
            + ", ".join(f"[{r.idx}] {r.step.name}" for r in rs)
        )

    raise OutputConflictError("\n".join(lines))


def _select_non_conflicting(runnable: list[_Runnable]) -> list[_Runnable]:
    """Greedy selection (stable by step order) of disjoint outputs."""
    selected: list[_Runnable] = []
    claimed: set[str] = set()
    for r in sorted(runnable, key=lambda x: x.idx):
        outs = set(r.step.outputs)
        if claimed.intersection(outs):
            continue
        selected.append(r)
        claimed.update(outs)
    return selected


def _execute_serial(data_store: Bundle, remaining: list[tuple[int, Step]]) -> Bundle:
    while remaining:
        available = set(data_store.keys())
        runnable = _iter_runnable(remaining, available)

        if not runnable:
            raise DeadEndError(_format_dead_end(remaining, available))

        r = min(runnable, key=lambda x: x.idx)
        required = {k: data_store[k] for k in r.step.inputs}

        try:
            out_bundle = _execute_step(idx=r.idx, step=r.step, required_bundle=required)
        except Exception as e:  # noqa: BLE001
            raise PipelineExecutionError(
                f"Step [{r.idx}] {r.step.name} failed: {e}"
            ) from e

        data_store.update(out_bundle)
        remaining = [(i, s) for (i, s) in remaining if i != r.idx]

    return data_store


@dataclass(frozen=True, slots=True)
class _ExecutorConfig:
    executor: Executor
    require_spec: bool


def _make_executor(backend: Backend, max_workers: int | None) -> _ExecutorConfig:
    if backend == "processes":
        return _ExecutorConfig(
            executor=ProcessPoolExecutor(max_workers=max_workers),
            require_spec=True,
        )
    if backend == "threads":
        return _ExecutorConfig(
            executor=ThreadPoolExecutor(max_workers=max_workers),
            require_spec=False,
        )
    raise ValueError(f"Unknown backend {backend!r}")


def _submit_step(
    *,
    ex: Executor,
    require_spec: bool,
    data_store: Bundle,
    r: _Runnable,
) -> Future[dict[str, OutputValue]]:
    required = {k: data_store[k] for k in r.step.inputs}

    if require_spec:
        for t in r.step.transforms:
            if not isinstance(t.target, str):
                raise PipelineExecutionError(
                    f"Step [{r.idx}] {r.step.name} must use import-string TransformSpec targets for processes backend"
                )

    return ex.submit(_execute_step, idx=r.idx, step=r.step, required_bundle=required)


def _collect_completed(
    *,
    pipeline: Pipeline,
    futures: dict[int, Future[dict[str, OutputValue]]],
) -> list[tuple[int, dict[str, OutputValue]]]:
    completed: list[tuple[int, dict[str, OutputValue]]] = []
    for idx, fut in futures.items():
        step_name = pipeline.steps[idx].name
        try:
            out_bundle = fut.result()
        except Exception as e:  # noqa: BLE001
            raise PipelineExecutionError(f"Step [{idx}] {step_name} failed: {e}") from e
        completed.append((idx, out_bundle))
    return completed


def _execute_parallel(
    *,
    pipeline: Pipeline,
    data_store: Bundle,
    remaining: list[tuple[int, Step]],
    backend: Backend,
    max_workers: int | None,
) -> Bundle:
    cfg = _make_executor(backend, max_workers)
    with cfg.executor as ex:
        while remaining:
            available = set(data_store.keys())
            runnable = _iter_runnable(remaining, available)

            if not runnable:
                raise DeadEndError(_format_dead_end(remaining, available))

            _detect_runnable_output_conflicts(runnable)

            batch = _select_non_conflicting(runnable)
            futures: dict[int, Future[dict[str, OutputValue]]] = {}
            for r in batch:
                futures[r.idx] = _submit_step(
                    ex=ex, require_spec=cfg.require_spec, data_store=data_store, r=r
                )

            completed = _collect_completed(pipeline=pipeline, futures=futures)
            for idx, out_bundle in sorted(completed, key=lambda x: x[0]):
                data_store.update(out_bundle)
                remaining = [(i, s) for (i, s) in remaining if i != idx]

    return data_store


def execute(
    pipeline: Pipeline,
    initial_bundle: Bundle,
    *,
    parallel: bool = False,
    backend: Backend = "processes",
    max_workers: int | None = None,
) -> Bundle:
    """Execute a pipeline from an initial named bundle.

    Args:
        pipeline: Pipeline configuration (ordered steps).
        initial_bundle: Initial data store. Keys are signal names; values are TimeSeries.
        parallel: If True, attempt to run independent steps concurrently.
        backend: Parallel backend when parallel=True ("processes" or "threads").
        max_workers: Passed through to the executor.

    Returns:
        Final data store (copy of initial with all produced outputs committed).

    Raises:
        OutputConflictError: If multiple runnable steps would write the same output key.
        DeadEndError: If the pipeline gets stuck due to missing inputs or cycles.
        PipelineExecutionError: For other execution issues.
    """

    data_store: Bundle = dict(initial_bundle)
    remaining: list[tuple[int, Step]] = list(enumerate(pipeline.steps))

    if not parallel:
        return _execute_serial(data_store, remaining)

    return _execute_parallel(
        pipeline=pipeline,
        data_store=data_store,
        remaining=remaining,
        backend=backend,
        max_workers=max_workers,
    )
