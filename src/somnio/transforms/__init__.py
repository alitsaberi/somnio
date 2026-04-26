"""Generic TimeSeries → TimeSeries transform functions.

These transforms are reusable building blocks that operate on
:class:`~somnio.data.timeseries.TimeSeries` objects and are independent of
any particular domain task.  Pipeline-compatible Bundle → Bundle wrappers are
also provided so that transforms can be referenced by import string in YAML
pipeline definitions.
"""
