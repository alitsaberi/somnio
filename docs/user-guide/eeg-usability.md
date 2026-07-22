# EEG usability

`somnio.tasks.eeg_usability` scores sleep EEG recordings for data usability —
detecting artifact-contaminated segments such as high noise, spiky noise, and
M-shaped noise. It wraps the **eegUsability lite** machine-learning models from
[eegFloss](https://github.com/Niloy333/eegFloss) (spectrogram features only).

Somnio reimplements the eegFloss lite feature-extraction and scoring pipeline as
a library API around :class:`~somnio.data.timeseries.TimeSeries`. Pre-trained
model weights are downloaded on first use from the same distribution used by
eegFloss.

## Install

```bash
pip install 'somnio[eeg-usability]'
```

Or with uv:

```bash
uv add somnio --extra eeg-usability
```

## Quick start

Input must be **256 Hz** and include EEG plus a movement channel. Data are
scored in **10-second epochs** (0.1 Hz output).

```python
from somnio.tasks.eeg_usability import get_usability_scores, load_model

model = load_model("lite")  # downloads on first use

scores, samples_to_keep, epoch_length = get_usability_scores(
    ts,
    model,
    eeg_left="EEG_L",
    eeg_right="EEG_R",
    movement="MOVEMENT",
)
```

For a single electrode, use :func:`~somnio.tasks.eeg_usability.get_usability_score`.

## Models

Only spectrogram-based (*lite*) models are supported:

| Version key | eegFloss name | Notes |
|---|---|---|
| `lite` (default) | v0.7 | Multiclass usability labels; recommended |
| `lite_binary` | v0.7.3 | Usable vs. not usable |

Pass the version key to :func:`~somnio.tasks.eeg_usability.load_model`.

Full eegFloss models that require TSFEL statistical features are not supported.

## Attribution

The eegUsability models and methodology were developed as part of
[eegFloss](https://github.com/Niloy333/eegFloss) (MIT License, Copyright
© 2025 Niloy Sikder).

If you use this functionality in academic work, please cite:

**Paper**

```bibtex
@article{sikder2025eegfloss,
  title     = {eegFloss: A Python package for refining sleep EEG recordings using machine learning models},
  author    = {Sikder, Niloy and Zerr, Paul and Jafarzadeh Esfahani, Mahdad and Dresler, Martin and Krauledat, Matthias},
  journal   = {arXiv preprint arXiv:2507.06433},
  year      = {2025},
  doi       = {10.48550/arXiv.2507.06433},
  url       = {https://arxiv.org/abs/2507.06433},
}
```

**Software**

```bibtex
@software{sikder2025eegflossv1,
  author    = {Niloy Sikder},
  title     = {eegFloss},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.15823969},
  url       = {https://doi.org/10.5281/zenodo.15823969},
}
```

Example methods text:

> EEG usability scores were computed with the eegUsability lite models from
> eegFloss (Sikder et al., 2025), accessed via the Somnio Python library.

## Further reading

- [eegFloss repository](https://github.com/Niloy333/eegFloss)
- [Paper (arXiv)](https://doi.org/10.48550/arXiv.2507.06433)
- [Software release (Zenodo)](https://doi.org/10.5281/zenodo.15823969)
