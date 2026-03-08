# CLI

The ZUtils CLI is the main way to use ZUtils from the command line. It requires the optional dependency group: install with `zutils[cli]` (see [Installation](../getting-started.md)).

## Usage

```bash
zutils [OPTIONS] COMMAND [ARGS]...
```

Running `zutils` with no arguments (or with `--help`) shows available commands. **Global options** must appear **before** the command name. See [CLI](../reference/cli.md) for the full reference.

## Shell completion

To install shell completion for the current shell:

```bash
zutils --install-completion
```

To show the completion script without installing it:

```bash
zutils --show-completion
```