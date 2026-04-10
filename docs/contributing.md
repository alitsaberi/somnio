# Contributing to Somnio

Thanks for helping improve **Somnio**. Contributions are welcome, including:

- Code changes via pull requests
- Documentation improvements
- Bug reports and feature requests

## Setup

### Prerequisites

- **Python**: Somnio supports Python **3.10+**; for local development we recommend **Python 3.12+**.
- **uv**: Install the [uv](https://docs.astral.sh/uv/) package manager.

### Installation

1\. Clone the repository

```bash
git clone https://github.com/alitsaberi/somnio.git
cd somnio
```

2\. Install dependencies

```bash
uv sync --all-extras --dev
```

3\. Set up pre-commit hooks
```bash
uv run pre-commit install
```

## Development workflow

This project follows [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow).

### Tips and Conventions

- Commit messages should follow the format **`Tag(context): Message`**.
    - **Tags**:
        - **`feat`**: New user-facing functionality
        - **`fix`**: Bug fix
        - **`docs`**: Documentation-only changes
        - **`refactor`**: Code change that neither fixes a bug nor adds a feature
        - **`test`**: Tests only
        - **`chore`**: Tooling/maintenance (deps, CI, formatting, etc.)
    - **Context**: A short scope for *where* the change applies (a package/module, feature area, or subsystem), e.g. `cli`, `zmax`, `docs`, `io`, `logging`.
    - **Examples**: `fix(zmax): Handle empty config`, `docs(contributing): Document commit format`, `feat(cli): Add --json output`
- Branch names should also use the same **tags**, with no context: **`tag/short-kebab-description`**, e.g. `fix/handle-empty-config`, `docs/contributing-commit-format`, `feat/cli-json-output`.
- Keep PRs focused; avoid bundling unrelated changes. Squash before merging.
- Rebase your branch on the latest `master` when needed to minimize merge conflicts.
- Make PR titles/descriptions clear and include a brief test plan when relevant.

## Code structure

Somnio is both a **library** (importable by other Python code) and a **CLI** application.

- **Library code**: `src/somnio/`
    - Library code should avoid CLI-only dependencies.
- **CLI code**: `src/somnio/cli/`
    - `src/somnio/cli/main.py`: Typer `app` and top-level options
    - `src/somnio/cli/commands/`: one module per subcommand

## Logging

To avoid surprising host applications:

- **Library code uses the standard library `logging` module** (no Loguru imports).
- **Library code must not configure handlers** (no `basicConfig()`, no global setup).
- The package installs a `NullHandler()` so importing `somnio` never emits logs by default.
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

Somnio uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting. If you've set up pre-commit hooks, linting and formatting will run automatically before each commit.

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

Somnio uses [pytest](https://docs.pytest.org/) for tests.

### Run tests

```bash
uv run pytest
```

### Test conventions

- **Location**: Put tests in `tests/`. Prefer mirroring the source layout (e.g., `src/somnio/foo.py` → `tests/test_foo.py`).
- **Naming**: Use `test_*.py` files and `test_*` functions. Name tests by behavior (`test_parses_empty_file`) rather than implementation.
- **Structure**: Use **Arrange–Act–Assert** and keep each test focused on one behavior.
- **Fixtures**: Prefer `pytest` fixtures for setup/teardown (and reuse). Avoid shared global state between tests.
- **Isolation**: Tests should be deterministic and run in any order. Avoid network calls and reliance on the local machine/user environment.
- **I/O**: Use `tmp_path` for filesystem work; don’t write into the repo tree.
- **Assertions**: Prefer simple `assert` statements; use `pytest.raises(...)` for error cases.
- **Speed**: Keep unit tests fast; if you add slow/integration tests, document how to run them and register any custom markers in `pyproject.toml`.

## Releases

Somnio uses **semantic versioning** (`MAJOR.MINOR.PATCH`). Breaking API or behavior changes increment **MAJOR**; backward-compatible additions increment **MINOR**; backward-compatible fixes increment **PATCH**. Pre-releases may use suffixes such as `a1`, `b1`, `rc1` on the version string in `pyproject.toml`.

### Changelog

User-facing changes for each version belong in the root [**CHANGELOG.md**](https://github.com/alitsaberi/somnio/blob/master/CHANGELOG.md). Before tagging a release, add a dated section for the new version (move items out of **Unreleased** when you cut the release).

### Git tags and PyPI

1. Set `version` in `pyproject.toml` to the release you are shipping (or run `uv version <x.y.z>`).
2. Commit the version bump and changelog updates on `master` (or your release branch, then merge).
3. Create an **annotated** tag whose name is `v` plus the same version string, for example `v0.2.0`:

   ```bash
   git tag -a v0.2.0 -m "v0.2.0"
   git push origin v0.2.0
   ```

   The tag name without the leading `v` **must** match `project.version` in `pyproject.toml`, or the [Publish to PyPI](https://github.com/alitsaberi/somnio/blob/master/.github/workflows/publish-pypi.yml) workflow will fail.

4. [GitHub Actions](https://docs.github.com/en/actions) builds the sdist and wheel, runs tests, and runs `uv publish` to **PyPI** using **Trusted Publishing** (OIDC). No long-lived API token is stored in the repository; configure the trusted publisher and environment once as described in the workflow file comments.

### GitHub Release notes

When you publish a tag, create a [**GitHub Release**](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) for that tag. Use the release description as the human-readable announcement: you can paste or summarize the corresponding **CHANGELOG.md** section so subscribers see highlights in their feed.

## Documentation

This repo uses MkDocs for documentation.

Serve docs locally:

```bash
uv run mkdocs serve
```

### Docstrings

All public modules, functions, classes, and methods should include docstrings. Somnio follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstring formatting.

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

Somnio uses a lightweight convention for TODOs so that follow-up work (docs, tests, refactors, etc.) stays visible and actionable.

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

