"""Execution engine for `somnio.pipeline`."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from somnio.data import TimeSeries
from somnio.pipeline.config import Backend, Bundle, Pipeline, Step, TransformSpec
from somnio.pipeline.errors import (
    DeadEndError,
    OutputConflictError,
    PipelineExecutionError,
)


def _resolve_import_string(target: str) -> Callable[..., Any]:
    if ":" not in target:
        raise ValueError(
            f"Invalid import string {target!r}; expected format 'pkg.module:callable'"
        )
    module_name, attr = target.split(":", 1)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, attr, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Import string {target!r} did not resolve to a callable")
    return fn


def _run_step_in_worker(
    spec: TransformSpec,
    required_bundle: dict[str, TimeSeries],
) -> dict[str, TimeSeries]:
    fn = _resolve_import_string(spec.target)
    return fn(required_bundle, **spec.kwargs)


def _run_step_inline(
    transform: TransformSpec | Callable[[Bundle], Bundle],
    required_bundle: dict[str, TimeSeries],
) -> dict[str, TimeSeries]:
    if isinstance(transform, TransformSpec):
        fn = _resolve_import_string(transform.target)
        return fn(required_bundle, **transform.kwargs)
    return transform(required_bundle)


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
        while remaining:
            available = set(data_store.keys())
            runnable = _iter_runnable(remaining, available)
            if not runnable:
                raise DeadEndError(_format_dead_end(remaining, available))
            _detect_runnable_output_conflicts(runnable)
            r = min(runnable, key=lambda x: x.idx)
            required = {k: data_store[k] for k in r.step.inputs}
            try:
                out_bundle = _run_step_inline(r.step.transform, required)
            except Exception as e:  # noqa: BLE001
                raise PipelineExecutionError(
                    f"Step [{r.idx}] {r.step.name} failed: {e}"
                ) from e
            if not isinstance(out_bundle, dict):
                raise PipelineExecutionError(
                    f"Step [{r.idx}] {r.step.name} returned {type(out_bundle)!r}, expected dict"
                )
            data_store.update(out_bundle)
            remaining = [(i, s) for (i, s) in remaining if i != r.idx]
        return data_store

    if backend == "processes":
        require_spec = True
        executor: Executor = ProcessPoolExecutor(max_workers=max_workers)
    elif backend == "threads":
        require_spec = False
        executor = ThreadPoolExecutor(max_workers=max_workers)
    else:
        raise ValueError(f"Unknown backend {backend!r}")

    with executor as ex:
        while remaining:
            available = set(data_store.keys())
            runnable = _iter_runnable(remaining, available)
            if not runnable:
                raise DeadEndError(_format_dead_end(remaining, available))
            _detect_runnable_output_conflicts(runnable)

            batch = _select_non_conflicting(runnable)
            futures: dict[int, Future[dict[str, TimeSeries]]] = {}
            for r in batch:
                required = {k: data_store[k] for k in r.step.inputs}
                if require_spec and not isinstance(r.step.transform, TransformSpec):
                    raise PipelineExecutionError(
                        f"Step [{r.idx}] {r.step.name} must use TransformSpec for processes backend"
                    )
                if isinstance(r.step.transform, TransformSpec):
                    futures[r.idx] = ex.submit(
                        _run_step_in_worker, r.step.transform, required
                    )
                else:
                    futures[r.idx] = ex.submit(r.step.transform, required)

            completed: list[tuple[int, dict[str, TimeSeries]]] = []
            for idx, fut in futures.items():
                step_name = pipeline.steps[idx].name
                try:
                    out_bundle = fut.result()
                except Exception as e:  # noqa: BLE001
                    raise PipelineExecutionError(
                        f"Step [{idx}] {step_name} failed: {e}"
                    ) from e
                if not isinstance(out_bundle, dict):
                    raise PipelineExecutionError(
                        f"Step [{idx}] {step_name} returned {type(out_bundle)!r}, expected dict"
                    )
                completed.append((idx, out_bundle))

            for idx, out_bundle in sorted(completed, key=lambda x: x[0]):
                data_store.update(out_bundle)
                remaining = [(i, s) for (i, s) in remaining if i != idx]

    return data_store
