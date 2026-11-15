# splinepy C++ core

The C++17 core provides spline evaluation, manipulation, Bézier algebra,
and proximity queries. Python uses the same kernels through nanobind.

## Build and test without Python bindings

Run these commands from the repository root:

```bash
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release -DSPLINEPY_COMPILE_PYTHON=OFF -DSPLINEPY_CORE_BUILD_TESTS=ON
cmake --build build-core --parallel
```

Some tests use golden fixtures. In an isolated Python 3.12 environment,
generate them with the released oracle before running CTest:

```bash
python -m pip install "numpy==2.2.6" "splinepy==0.2.1"
python -I tests/generate_golden_data.py
ctest --test-dir build-core --output-on-failure --no-tests=error
```
