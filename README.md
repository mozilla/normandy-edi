# normandy-edi

A CLI for working with the Normandy API.

## Usage

Edi can currently only be used from a development environment. To manage Python requirements, [Poetry](https://python-poetry.org) is used. After installing Poetry, run the following command to install the requirements for Edi:

```bash
$ poetry install
```

After this, to run Edi

```bash
$ poetry run edi
```

When run without arguments, or with not enough, Edi will print a help message detailing what further options you can use. For example:

```bash
$ poetry run edi recipes

Usage: edi recipes [OPTIONS] COMMAND [ARGS]...

Options:
  -v, --verbose
  -s, --server [prod|stage|dev|local|prod-admin|stage-admin|dev-admin]
  --help                          Show this message and exit.

Commands:
  all
  classify-filters
  count-filters
  delete
  empty
  get
  revise
  summarize         Show recipes enabled during a time range
```

## Examples

### The name of all heartbeat recipes that were active in 2019

```bash
$ poetry run edi recipes all \
    --action show-heartbeat \
    --enabled-begin 2019-01-01 \
    --enabled-end 2020-01-01 \
    --jq-query '.[].latest_revision.name'
```
