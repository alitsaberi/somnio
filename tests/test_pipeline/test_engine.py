import time

import numpy as np
import pytest

from somnio.data import TimeSeries
from somnio.pipeline import (
    DeadEndError,
    OutputConflictError,
    Pipeline,
    Step,
    TransformSpec,
    execute,
)


def _make_ts(value: float, *, t0: int = 0, n: int = 4) -> TimeSeries:
    return TimeSeries(
        values=np.full((n, 1), value, dtype=np.float64),
        timestamps=np.arange(t0, t0 + n, dtype=np.int64),
        channel_names=("X",),
        units=("V",),
        sample_rate=None,
    )


def add_constant(
    bundle: dict[str, TimeSeries], *, key: str, constant: float
) -> dict[str, TimeSeries]:
    ts = bundle[key]
    out = TimeSeries(
        values=ts.values + constant,
        timestamps=ts.timestamps.copy(),
        channel_names=ts.channel_names,
        units=ts.units,
        sample_rate=ts.sample_rate,
    )
    return {key: out}


def copy_to(
    bundle: dict[str, TimeSeries], *, src: str, dst: str
) -> dict[str, TimeSeries]:
    ts = bundle[src]
    return {dst: ts}


def sum_two(
    bundle: dict[str, TimeSeries], *, a: str, b: str, out: str
) -> dict[str, TimeSeries]:
    ta = bundle[a]
    tb = bundle[b]
    assert ta.timestamps.shape == tb.timestamps.shape
    out_ts = TimeSeries(
        values=ta.values + tb.values,
        timestamps=ta.timestamps.copy(),
        channel_names=ta.channel_names,
        units=ta.units,
        sample_rate=None,
    )
    return {out: out_ts}


def sleep_then_copy(
    bundle: dict[str, TimeSeries], *, src: str, dst: str, seconds: float
) -> dict[str, TimeSeries]:
    time.sleep(seconds)
    return copy_to(bundle, src=src, dst=dst)


def assert_only_keys_then_copy(
    bundle: dict[str, TimeSeries], *, expected: tuple[str, ...], src: str, dst: str
) -> dict[str, TimeSeries]:
    assert set(bundle.keys()) == set(expected)
    return copy_to(bundle, src=src, dst=dst)


def test_fan_out_then_downstream_dependency_serial() -> None:
    p = Pipeline.from_steps(
        [
            Step(
                name="fanout_a",
                inputs=("x",),
                outputs=("a",),
                transform=TransformSpec(
                    __name__ + ":copy_to", {"src": "x", "dst": "a"}
                ),
            ),
            Step(
                name="fanout_b",
                inputs=("x",),
                outputs=("b",),
                transform=TransformSpec(
                    __name__ + ":copy_to", {"src": "x", "dst": "b"}
                ),
            ),
            Step(
                name="join",
                inputs=("a", "b"),
                outputs=("y",),
                transform=TransformSpec(
                    __name__ + ":sum_two", {"a": "a", "b": "b", "out": "y"}
                ),
            ),
        ]
    )
    out = execute(p, {"x": _make_ts(2.0)}, parallel=False)
    assert "y" in out
    np.testing.assert_allclose(out["y"].values, 4.0)


def test_parallel_threads_runs_concurrently() -> None:
    p = Pipeline.from_steps(
        [
            Step(
                name="sleep_copy_a",
                inputs=("x",),
                outputs=("a",),
                transform=TransformSpec(
                    __name__ + ":sleep_then_copy",
                    {"src": "x", "dst": "a", "seconds": 0.25},
                ),
            ),
            Step(
                name="sleep_copy_b",
                inputs=("x",),
                outputs=("b",),
                transform=TransformSpec(
                    __name__ + ":sleep_then_copy",
                    {"src": "x", "dst": "b", "seconds": 0.25},
                ),
            ),
        ]
    )
    t0 = time.perf_counter()
    out = execute(
        p, {"x": _make_ts(1.0)}, parallel=True, backend="threads", max_workers=2
    )
    elapsed = time.perf_counter() - t0
    assert "a" in out and "b" in out
    # Two 0.25s steps should overlap; allow some overhead.
    assert elapsed < 0.45


def test_output_conflict_detection_for_runnable_steps() -> None:
    p = Pipeline.from_steps(
        [
            Step(
                name="write_same_1",
                inputs=("x",),
                outputs=("y",),
                transform=TransformSpec(
                    __name__ + ":copy_to", {"src": "x", "dst": "y"}
                ),
            ),
            Step(
                name="write_same_2",
                inputs=("x",),
                outputs=("y",),
                transform=TransformSpec(
                    __name__ + ":add_constant", {"key": "x", "constant": 1.0}
                ),
            ),
        ]
    )
    with pytest.raises(OutputConflictError, match="output 'y'"):
        _ = execute(p, {"x": _make_ts(1.0)}, parallel=True, backend="threads")


def test_dead_end_diagnostics() -> None:
    p = Pipeline.from_steps(
        [
            Step(
                name="needs_missing",
                inputs=("missing",),
                outputs=("y",),
                transform=TransformSpec(
                    __name__ + ":copy_to", {"src": "missing", "dst": "y"}
                ),
            )
        ]
    )
    with pytest.raises(DeadEndError, match="missing"):
        _ = execute(p, {"x": _make_ts(1.0)}, parallel=False)


def test_bundle_contract_only_required_inputs_passed() -> None:
    p = Pipeline.from_steps(
        [
            Step(
                name="copy_only_x",
                inputs=("x",),
                outputs=("y",),
                transform=TransformSpec(
                    __name__ + ":assert_only_keys_then_copy",
                    {"expected": ("x",), "src": "x", "dst": "y"},
                ),
            ),
        ]
    )
    out = execute(p, {"x": _make_ts(1.0), "extra": _make_ts(9.0)}, parallel=False)
    assert "y" in out


def test_bundle_contract_allows_heterogeneous_sample_rates() -> None:
    # Two independent signals with different timestamp grids / nominal rates.
    x = _make_ts(1.0, t0=0, n=4)
    y = _make_ts(2.0, t0=10, n=7)
    x.sample_rate = 100.0
    y.sample_rate = 25.0

    p = Pipeline.from_steps(
        [
            Step(
                name="copy_x",
                inputs=("x",),
                outputs=("x2",),
                transform=TransformSpec(
                    __name__ + ":copy_to", {"src": "x", "dst": "x2"}
                ),
            ),
            Step(
                name="copy_y",
                inputs=("y",),
                outputs=("y2",),
                transform=TransformSpec(
                    __name__ + ":copy_to", {"src": "y", "dst": "y2"}
                ),
            ),
        ]
    )
    out = execute(p, {"x": x, "y": y}, parallel=True, backend="threads", max_workers=2)
    assert out["x2"].timestamps.shape == (4,)
    assert out["y2"].timestamps.shape == (7,)
    assert out["x2"].sample_rate == 100.0
    assert out["y2"].sample_rate == 25.0


def test_parallel_processes_smoke_with_import_string_transform() -> None:
    p = Pipeline.from_steps(
        [
            Step(
                name="copy_x",
                inputs=("x",),
                outputs=("y",),
                transform=TransformSpec(
                    __name__ + ":copy_to", {"src": "x", "dst": "y"}
                ),
            )
        ]
    )
    out = execute(
        p, {"x": _make_ts(3.0)}, parallel=True, backend="processes", max_workers=1
    )
    assert "y" in out
    np.testing.assert_allclose(out["y"].values, 3.0)
