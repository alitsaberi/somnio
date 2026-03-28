# CLI

The Somnio CLI is the main way to use Somnio from the command line. It requires the optional dependency group: install with `somnio[cli]` (see [Installation](../getting-started.md)).

## Usage

```bash
somnio [OPTIONS] COMMAND [ARGS]...
```

Running `somnio` with no arguments (or with `--help`) shows available commands. **Global options** must appear **before** the command name. See [CLI](../reference/cli.md) for the full reference.

## Shell completion

To install shell completion for the current shell:

```bash
somnio --install-completion
```

To show the completion script without installing it:

```bash
somnio --show-completion
```