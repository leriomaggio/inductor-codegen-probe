# inductor-codegen-probe

A small self-contained probe built around a single empirical question. One
script, no framework.

## The question

When you run `torch.compile` with its default backend (Inductor), does Inductor
emit Triton, and does the answer change between CPU and MPS on Apple Silicon?

The script does not assume an answer. It compiles two trivial functions, runs
them on each available device to force compilation, captures the code Inductor
generated, and classifies it as one of: Triton, C++/OpenMP, Metal/MPS, an
extern-kernel dispatch (no Inductor codegen), or fallback.

## What it does

1. Prints environment facts: `torch.__version__`,
   `torch.backends.mps.is_available()`, and the default `torch.compile` backend.
2. Defines two functions: a matmul (`a @ b`) and a pointwise fusion
   (`(a * b + c).relu()`).
3. For each available device in `["cpu", "mps"]` and each function, it compiles
   the function, runs it once on that device to trigger compilation, captures the
   generated code, and classifies the codegen path. A missing device is skipped
   without failing.
4. Prints a `device x function -> codegen kind` summary table and an excerpt of
   each generated kernel. The excerpt skips the import boilerplate that every
   wrapper module repeats and shows the kernel body itself.

Code is captured with `torch._inductor.utils.run_and_get_triton_code`. If that
name has moved or errors out, the script falls back to `run_and_get_code`, and
then to capturing the `TORCH_LOGS=output_code` logging output. The helper has
changed location across torch releases, so all three paths are attempted before
giving up.

## How to run

```bash
uv run python probe.py
```

You can also run `python probe.py` from inside the project virtualenv. The only
dependency is torch.

The CPU path runs first and is known to work. The MPS path is wrapped so that a
failure there prints the error and lets the rest of the run continue.

## Results

Recorded on Apple Silicon with torch 2.12.1 and MPS available.

| device | function                   | codegen kind                        |
|--------|----------------------------|-------------------------------------|
| cpu    | matmul                     | extern kernel (no Inductor codegen) |
| cpu    | pointwise `(a*b+c).relu()` | C++ / OpenMP                        |
| mps    | matmul                     | extern kernel (no Inductor codegen) |
| mps    | pointwise `(a*b+c).relu()` | Metal / MPS                         |

On this machine MPS is not a Triton codegen target. Where Inductor actually
generates a kernel, which here is the fusible pointwise op, it emits a Metal
shader on MPS (`async_compile.metal(...)`, `kernel void ...`) and vectorized C++
on CPU (`cpp_fused_...`, `at::vec::`). Neither device emits Triton for these ops.

The matmul is a separate case. On both devices Inductor generates no kernel of
its own and dispatches to an external library op through
`extern_kernels.mm(...)`, which is BLAS on CPU and the MPS matmul on MPS. Triton
codegen from Inductor is the CUDA GPU path, and it did not appear on either CPU
or MPS here.

Paste your own run below if you want to keep a record of it.

```
<paste your output here>
```
