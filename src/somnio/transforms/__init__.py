"""Generic TimeSeries → TimeSeries transform functions.

These transforms are reusable building blocks that operate on
:class:`~somnio.data.timeseries.TimeSeries` objects and are independent of
any particular domain task.

Note: these modules intentionally expose **TimeSeries → TimeSeries** helpers.
Pipeline-level Bundle → Bundle wrappers (when needed) should live in the
domain task packages (e.g. under `somnio.tasks.*`) so they can enforce
task-specific bundle keys and output contracts.
"""
