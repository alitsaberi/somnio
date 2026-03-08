# Contributing to ZUtils

Thanks for helping improve **ZUtils**. Contributions are welcome, including:

- Code changes via pull requests
- Documentation improvements
- Bug reports and feature requests

## Setup

### Prerequisites

- **Python**: ZUtils supports Python **3.10+**; for local development we recommend **Python 3.12+**.
- **uv**: Install the [uv](https://docs.astral.sh/uv/) package manager.

### Installation

1\. Clone the repository

```bash
git clone https://github.com/alitsaberi/zutils.git
cd zutils
```

2\. Install dependencies

```bash
uv sync
```

3\. Set up pre-commit hooks
```bash
uv run pre-commit install
```

## Development workflow

This project follows [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow).

### Tips and Conventions

- Commit messages should start with a **capitalized action verb** (imperative) and be descriptive, e.g. `Add logging for debug output`, `Fix crash on empty config`, `Update contributing guide`.
- Keep PRs focused; avoid bundling unrelated changes. Squash before merging.
- Rebase your branch on the latest `master` when needed to minimize merge conflicts.
- Make PR titles/descriptions clear and include a brief test plan when relevant.

## Code structure

ZUtils is both a **library** (importable by other Python code) and a **CLI** application.

- **Library code**: `src/zutils/`
    - Library code should avoid CLI-only dependencies.
- **CLI code**: `src/zutils/cli/`
    - `src/zutils/cli/main.py`: Typer `app` and top-level options
    - `src/zutils/cli/commands/`: one module per subcommand

## Logging

To avoid surprising host applications:

- **Library code uses the standard library `logging` module** (no Loguru imports).
- **Library code must not configure handlers** (no `basicConfig()`, no global setup).
- The package installs a `NullHandler()` so importing `zutils` never emits logs by default.
- The **CLI** uses [Loguru](https://github.com/Delgan/loguru) and configures logging once at startup.

### Writing logs in library modules

Use module-level loggers:

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Something happened")
```

### Writing logs in CLI code

Use Loguru logger:

```python
from loguru import logger

logger.info("Something happened")
```

## Linting and formatting

ZUtils uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting. If you've set up pre-commit hooks, linting and formatting will run automatically before each commit.

### Run the linter

```bash
uv run ruff check .
```

Fix issues automatically when possible:

```bash
uv run ruff check --fix .
```

### Format code

```bash
uv run ruff format .
```

### Pre-commit (manual run)

You can also run all pre-commit checks manually:

```bash
uv run pre-commit run --all-files
```

## Testing

ZUtils uses [pytest](https://docs.pytest.org/) for tests.

### Run tests

```bash
uv run pytest
```

### Test conventions

- **Location**: Put tests in `tests/`. Prefer mirroring the source layout (e.g., `src/zutils/foo.py` → `tests/test_foo.py`).
- **Naming**: Use `test_*.py` files and `test_*` functions. Name tests by behavior (`test_parses_empty_file`) rather than implementation.
- **Structure**: Use **Arrange–Act–Assert** and keep each test focused on one behavior.
- **Fixtures**: Prefer `pytest` fixtures for setup/teardown (and reuse). Avoid shared global state between tests.
- **Isolation**: Tests should be deterministic and run in any order. Avoid network calls and reliance on the local machine/user environment.
- **I/O**: Use `tmp_path` for filesystem work; don’t write into the repo tree.
- **Assertions**: Prefer simple `assert` statements; use `pytest.raises(...)` for error cases.
- **Speed**: Keep unit tests fast; if you add slow/integration tests, document how to run them and register any custom markers in `pyproject.toml`.

## Documentation

This repo uses MkDocs for documentation.

Serve docs locally:

```bash
uv run mkdocs serve
```

### Docstrings

All public modules, functions, classes, and methods should include docstrings. ZUtils follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstring formatting.

Where appropriate, API and reference documentation should be generated automatically using **MkDocstrings**. Auto-generated reference documentation is intended to complement narrative documentation, not replace it.

### Writing conventions

- Use **kebab-case** for file names (e.g., `installation-setup.md`)
- Place files in the appropriate subfolder under `docs/`
- Use a single top-level heading (`#`) per page
- Use `##`, `###`, etc. for subsections
- Use **relative links** when linking between documentation pages
- Keep paths stable to avoid broken links
- Use fenced code blocks with a language identifier where possible:

```python
def example_function():
    pass
```

- Keep pages focused and concise
- Prefer multiple small pages over one very long page
- Write in clear, direct language

## TODOs and issue tracking

ZUtils uses a lightweight convention for TODOs so that follow-up work (docs, tests, refactors, etc.) stays visible and actionable.

- **Source of truth**: Anything non-trivial should be tracked as a **GitHub issue** (use the issue templates when possible).
- **In-code TODOs**: Allowed. Prefer linking to an issue/ticket for anything that won't be handled immediately.
- **Author is optional**: You may write `TODO:` or `TODO(name):`.

### TODO format (Python)

Prefer one of these forms:

```python
# TODO: [tests] Add regression test for empty config.

# TODO(ali): [docs] Explain config precedence.
```

Notes:

- Use a short tag like `[docs]`, `[tests]`, `[refactor]`, `[logging]`, `[config]` at the start of the description.
- If applicable, add an issue reference on the next line (URL) or a ticket id like `ABC-123`.

### TODO format (Markdown)

Use HTML comments so TODOs don't render in the docs:

```md
<!-- TODO: [docs] Add example CLI usage. -->
```

### Suggested labels

If you maintain labels in the GitHub repo, these tend to work well:

- **Area**: `docs`, `tests`, `cli`, `logging`, `config`
- **Type**: `bug`, `tech-debt`
- **Priority**: `priority:high`, `priority:medium`, `priority:low`

