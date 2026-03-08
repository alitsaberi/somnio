# Download from NSRR

Download files from a dataset on the [National Sleep Research Resource (NSRR)](https://sleepdata.org/). 

## Synopsis

```bash
zutils download-nsrr SLUG OUTPUT_DIR [OPTIONS]
```

See [CLI](../reference/cli.md) for the full reference.

## Before you start

### Get an NSRR token

- Create an account at [sleepdata.org](https://sleepdata.org/).
- Request access to the dataset (e.g. SOF, SHHS, MESA).
- After approval, copy your **auth token** from the dataset page or account profile.
- Pass it with `--token` or set the `NSRR_TOKEN` environment variable (e.g. in `.env`).

## Output layout

Files are placed under `OUTPUT_DIR/SLUG/`. If you use `--path`, only that subpath is created. For example:

- `zutils download-nsrr sof ./data ...` → `./data/sof/...`
- `zutils download-nsrr shhs ./data --path polysomnography ...` → `./data/shhs/polysomnography/...`

## Examples

### Full dataset

Download the full **SOF** dataset into `./data`:

```bash
zutils download-nsrr sof ./data --token YOUR_NSRR_TOKEN
```

### Single subpath

Download only the `polysomnography` folder from **SHHS**:

```bash
zutils download-nsrr shhs ./data --path polysomnography --token YOUR_NSRR_TOKEN
```

### Using the environment for the token

Avoid putting the token on the command line by setting `NSRR_TOKEN` (e.g. in `.env` or your shell):

```bash
export NSRR_TOKEN=your_token_here
zutils download-nsrr sof ./data
```

### Longer timeout and file logging

Use a 10-minute timeout and write logs to the default `logs/` file:

```bash
zutils -l download-nsrr mesa ./datasets --token YOUR_NSRR_TOKEN --timeout-seconds 600
```

### Custom log file

Send debug logs to a specific file for later inspection:

```bash
zutils --log-file ./nsrr-debug.log download-nsrr shhs ./data --token YOUR_NSRR_TOKEN
```

### Retries and timeouts

For flaky connections or slow networks, use `--timeout-seconds` to increase the per-request timeout. You can also tune `--download-retries` (retries per file on connection/read timeout) and `--http-retries` (retries for API requests). See the [CLI reference](../reference/cli.md) for defaults and details.
