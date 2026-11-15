# Contributing
splinepy welcomes and appreciates discussions, issues and pull requests!

## Quick start
Fork and clone the repository, then create a branch for your changes. Use
[uv](https://docs.astral.sh/uv/) to set up the development environment and
install splinepy in editable mode.

```bash
git clone git@github.com:<path-to-your-fork>
cd splinepy
git switch -c my-feature
uv sync --locked --extra test
uv run --no-sync pytest
```

For a C++ Debug build, add `--config-setting cmake.build-type=Debug` to
`uv sync`.

## Python style / implementation preferences
- use [`PEP 8`](https://peps.python.org/pep-0008/) style guide
- no complex comphrehensions: preferably fits in a line, 2 lines max if it is totally necessary
- use first letter abbreviations in element loops:  `for kv in knot_vectors`
- use `i`, `j`, `k`, `l` for pure index: `for i, kv in enumerate(knot_vectors)`
- module local import with a leading underscore: `from splinepy import settings as _settings`
- make sure function wrappers are properly documented (see, e.g., helpme.create.Creator's member functions)

## C++ style / implementation preferences
For c++, we've prepared a `.clang-format`, with that you can just run `clang-format`. We closely follow naming scheme suggested by [google stype guide](https://google.github.io/styleguide/cppguide.html#Naming), with a clear exception of file naming: we use [`.hpp`, `.inl`, `.cpp`].
Here's another preference:
- `#pragma once`

## Automatic formatting / style check
Run all formatting and lint checks from the repository root.

```bash
uv run --no-sync --with pre-commit pre-commit run -a
```

## Local documentation build

Install Doxygen and Graphviz, then run these commands from the repository root.

```bash
uv sync --locked --extra gustaf --extra viewer
(cd docs && mkdir -p build && doxygen Doxyfile)
uv run --no-sync python docs/source/handle_markdown.py
uv run --no-sync python docs/markdown/extra_docs.py
uv run --no-sync --with-requirements docs/requirements.txt sphinx-build -W --keep-going -b html docs/source docs/build
```

Open `docs/build/index.html` for Sphinx or `docs/build/doxy/html/index.html`
for the C++ reference.
