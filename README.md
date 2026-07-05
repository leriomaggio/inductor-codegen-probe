# PyTorch CodeGen Probe

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

### Machine

The output below was recorded on this setup:

| item     | value                                     |
|----------|-------------------------------------------|
| model    | Apple Mac15,6 (MacBook Pro, Apple M3 Pro) |
| memory   | 36 GB                                     |
| arch     | arm64                                     |
| OS       | macOS 26.5.1 (build 25F80)                |
| Python   | 3.12.7                                     |
| torch    | 2.12.1                                     |
| MPS      | available                                 |

### Summary

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

### Full output

```
======================================================================
ENVIRONMENT
======================================================================
torch.__version__ ............. 2.12.1
torch.backends.mps.is_available() True
default torch.compile backend . 'inductor'
devices probed ................ ['cpu', 'mps']

----------------------------------------------------------------------
[cpu] matmul
----------------------------------------------------------------------
codegen kind : extern kernel (no Inductor codegen)
captured via : run_and_get_triton_code
generated kernel (excerpt):
(from line 49)
        assert_size_stride(arg1_1, (64, 64), (64, 1))
        buf0 = empty_strided_cpu((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
        extern_kernels.mm(arg0_1, arg1_1, out=buf0)
        del arg0_1
        del arg1_1
        return (buf0, )
... (more lines)

----------------------------------------------------------------------
[cpu] pointwise (a*b+c).relu()
----------------------------------------------------------------------
codegen kind : C++ / OpenMP
captured via : run_and_get_triton_code
generated kernel (excerpt):
(from line 32)
cpp_fused_add_mul_relu_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(1024LL); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(1024LL)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
... (more lines)

----------------------------------------------------------------------
[mps] matmul
----------------------------------------------------------------------
codegen kind : extern kernel (no Inductor codegen)
captured via : run_and_get_triton_code
generated kernel (excerpt):
(from line 52)
        arg1_1 = copy_misaligned(arg1_1)
        buf0 = empty_strided((64, 64), (64, 1), device='mps', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
        extern_kernels.mm(arg0_1, arg1_1, out=buf0)
        del arg0_1
        del arg1_1
        return (buf0, )
... (more lines)

----------------------------------------------------------------------
[mps] pointwise (a*b+c).relu()
----------------------------------------------------------------------
codegen kind : Metal / MPS
captured via : run_and_get_triton_code
generated kernel (excerpt):
(from line 31)
generated_kernel_0 = async_compile.metal('generated_kernel_0', '''

    kernel void generated_kernel_0(
        device float* out_ptr0,
        constant float* in_ptr0,
        constant float* in_ptr1,
        constant float* in_ptr2,
        uint xindex [[thread_position_in_grid]]
    ) {
        int x0 = xindex;
        auto tmp0 = in_ptr0[x0];
        auto tmp1 = in_ptr1[x0];
        auto tmp3 = in_ptr2[x0];
        auto tmp2 = tmp0 * tmp1;
        auto tmp4 = tmp2 + tmp3;
... (more lines)

======================================================================
SUMMARY:  device x function  ->  codegen kind
======================================================================
  cpu  |  matmul                    ->  extern kernel (no Inductor codegen)   (run_and_get_triton_code)
  cpu  |  pointwise (a*b+c).relu()  ->  C++ / OpenMP   (run_and_get_triton_code)
  mps  |  matmul                    ->  extern kernel (no Inductor codegen)   (run_and_get_triton_code)
  mps  |  pointwise (a*b+c).relu()  ->  Metal / MPS   (run_and_get_triton_code)

======================================================================
VERDICT
======================================================================
CPU codegen : {'matmul': 'extern kernel (no Inductor codegen)', 'pointwise (a*b+c).relu()': 'C++ / OpenMP'}
MPS codegen : {'matmul': 'extern kernel (no Inductor codegen)', 'pointwise (a*b+c).relu()': 'Metal / MPS'}

=> MPS is NOT a Triton codegen target. Where Inductor generated a kernel
   (the fusible pointwise op), it emitted: Metal / MPS, not Triton.
   Ops with no Inductor codegen (e.g. matmul) went to: extern kernel (no Inductor codegen).
```
