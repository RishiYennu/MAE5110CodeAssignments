# MAE 5110 Code Assignments

Code assignments for MAE 5110.

## Installation

Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/). After cloning this repository, run the following command from its root directory:

```console
uv sync --python 3.14
```

This creates a local `.venv`, installs the dependencies, and installs this repo in
[editable mode](https://docs.astral.sh/uv/concepts/projects/config/#editable-mode).
Imports use your working files, so code changes take effect the next time you run
them without reinstalling. Run Python commands inside the environment with `uv run`, for example:

```console
uv run python assignment_0.py
```

## Assignments

- [Assignment 0](assignments/assignment_0.md)
- [Assignment 1](assignments/assignment_1.md)
- [Assignment 2](assignments/assignment_2.md)
- [Assignment 3](assignments/assignment_3.md)
