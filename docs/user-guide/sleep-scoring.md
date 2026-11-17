# Sleep scoring

`somnio.tasks.sleep_scoring` runs sleep-stage inference on a
:class:`~somnio.data.timeseries.TimeSeries` using exported neural models
(ONNX Runtime or TensorFlow SavedModel) plus a `model.yaml` sidecar.

The public entry point is
:func:`~somnio.tasks.sleep_scoring.score_sleep_stages`. Backends implement
:class:`~somnio.tasks.sleep_scoring.SleepScoringBackend`.

## Install

Metadata validation needs the `schemas` extra. Pick a backend extra as well:

```bash
pip install 'somnio[schemas,onnx]'
# or
pip install 'somnio[schemas,tensorflow]'
```

With uv:

```bash
uv add somnio --extra schemas --extra onnx
# or
uv add somnio --extra schemas --extra tensorflow
```

The `tensorflow` extra installs TensorFlow only on Python versions before 3.14
(no cp314 wheels yet).

## Model artifacts

Expect an exported model directory or file layout like:

| Format | Weights | Metadata |
|---|---|---|
| ONNX | `model.onnx` (file) | Sibling `model.yaml` (or `{stem}.yaml`) |
| TensorFlow | SavedModel **directory** | `model.yaml` **inside** that directory |

Minimal sidecar fields:

```yaml
sample_rate_hz: 128.0
n_periods_per_window: 35   # or null for dynamic / one-shot
n_samples_per_period: 3840
n_channels: 1
n_samples_per_prediction: 3840
class_labels: [W, N1, N2, N3, REM]
```

- **Fixed window**: `n_periods_per_window` is a positive integer (sliding batches).
- **Dynamic / one-shot**: `n_periods_per_window` is `null`; the whole recording is
  scored in a single variable-length batch.

Optional `onnx` / `tensorflow` blocks can name graph I/O tensors; if omitted,
the first model input and output are used.

The `preprocessing` list in `model.yaml` is **reserved**. It is validated when
present but **not applied** by
:func:`~somnio.tasks.sleep_scoring.score_sleep_stages`. Preprocess the input
series yourself before scoring (for example with `somnio.transforms`).

## Quick start (ONNX)

```python
from somnio.tasks.sleep_scoring import score_sleep_stages
from somnio.tasks.sleep_scoring.models import OnnxSleepScoringModel

model = OnnxSleepScoringModel.load("path/to/model.onnx")  # finds model.yaml
probs = score_sleep_stages(
    ts,
    backend=model,
    metadata=model.metadata,
    output="probs_timeseries",
)
```

## Quick start (TensorFlow)

```python
from somnio.tasks.sleep_scoring import score_sleep_stages
from somnio.tasks.sleep_scoring.models import TensorflowSleepScoringModel

model = TensorflowSleepScoringModel.load("path/to/tf_saved_model_dir")
labels = score_sleep_stages(
    ts,
    backend=model,
    metadata=model.metadata,
    output="labels_epochs",
)
```

Pass an explicit metadata path or
:class:`~somnio.tasks.sleep_scoring.ModelMetadata` instance to `.load(...)` if
the sidecar is not next to the artifact.

## Outputs

`output` selects the return type:

| Value | Result |
|---|---|
| `probs_timeseries` (default) | Class-probability :class:`~somnio.data.timeseries.TimeSeries` at prediction resolution |
| `indices_epochs` | :class:`~somnio.data.annotations.Epochs` of argmax class indices |
| `labels_epochs` | :class:`~somnio.data.annotations.Epochs` of argmax class label strings |

Input `ts` should already match `metadata.sample_rate_hz` and
`metadata.n_channels`. Period stride defaults to non-overlapping periods
(`n_samples_per_period`).

## Further reading

- Package API: `somnio.tasks.sleep_scoring`
- Optional extras: [Getting started](../getting-started.md)
