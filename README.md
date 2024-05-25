# foundation

This repository serves as a template with configuration files for common python tools,
including `poetry`, `pre-commit`, `black`, `isort`, etc.

## Development setup

### Conda

Recommend to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a virtual environment:

```shell
$ conda create --name foundation python=3.9
# Answer yes to the prompt
$ conda activate foundation
```

Whenever you work on this you can activate the environment again using:

```shell
$ conda activate foundation
```

### Poetry

The project is set up using the [poetry](https://python-poetry.org/docs/) dependency management tool.

For Linux, macOS, and Windows (WSL):

```shell
$ curl -sSL https://install.python-poetry.org | python3 -
```

An alternative way for macOS is:

```shell
$ brew install poetry
```

### Installation

```shell
$ poetry install
# Initialize pre-commit hooks
$ pre-commit install
```

Run test cases with:

```shell
$ pytest -n 8
```

## References

Collect some useful code snippets from following repositories with nicely modification:

1. [open-mmlab/mmcv](https://github.com/open-mmlab/mmcv)
1. [facebookresearch/fvcore](https://github.com/facebookresearch/fvcore)
1. [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
