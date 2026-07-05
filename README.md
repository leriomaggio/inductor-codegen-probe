# PyTorch CodeGen Probe

A small self-contained probe into what `torch.compile` actually generates, and
how that changes across the compute backends present on a machine. One script,
no framework.

## The question

When you run `torch.compile` with its default backend (Inductor), what code does
Inductor generate, and how does the answer change across the devices available
here: CPU, CUDA, MPS on Apple Silicon, and Intel XPU?

The script does not assume an answer. It compiles a few trivial functions, runs
each on every available device to force compilation, captures the code Inductor
generated, and classifies it as one of: Triton, C++/OpenMP, Metal/MPS, an
extern-kernel dispatch (no Inductor codegen), or fallback.

## Why these functions

The three probes are chosen to exercise different codegen shapes, so the output
shows more than a single path:

- `matmul (a @ b)` is a library-shaped op. Inductor often hands it off wholesale
  to an external BLAS-style kernel rather than generating anything.
- `pointwise (a*b+c).relu()` is an elementwise chain that Inductor can fuse into
  one generated kernel.
- `reduction softmax(a, -1)` fuses reductions (max, sum) with pointwise work,
  which is where the device-specific kernel languages differ most visibly (for
  example OpenMP-parallel C++ with thread-local buffers on CPU, versus a Metal
  kernel using threadgroup reductions on MPS).

## What it does

1. Prints environment facts: `torch.__version__`, the default `torch.compile`
   backend, and which of CUDA / MPS / XPU report themselves available.
2. Discovers every backend it can probe. CPU is always present; each accelerator
   is added only when its runtime is available, so the same script runs unchanged
   on a CPU-only host, an Apple Silicon laptop, a CUDA box, or an Intel XPU host.
3. For each device and each function it compiles the function, runs it once on
   that device to trigger compilation, captures the generated code, and
   classifies the codegen path. A backend that fails is reported and skipped
   rather than aborting the run.
4. Prints a `device x function -> codegen kind` summary table, an excerpt of each
   generated kernel, and a neutral per-device findings block. No device is
   treated as the special case; each is reported on the same terms.

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

The CPU path always runs. Each accelerator path is wrapped so that a failure
there prints the error and lets the rest of the run continue.

## Results

The output below was recorded on one specific machine. A CUDA or XPU host would
add rows for those devices; the point of the script is that you run it on yours
and read the table it prints.

### Machine

| item   | value                                     |
|--------|-------------------------------------------|
| model  | Apple Mac15,6 (MacBook Pro, Apple M3 Pro) |
| memory | 36 GB                                     |
| arch   | arm64                                     |
| OS     | macOS 26.5.1 (build 25F80)                |
| Python | 3.12.7                                    |
| torch  | 2.12.1                                    |
| CUDA   | not available                             |
| MPS    | available                                 |
| XPU    | not available                             |

### Summary

| device | function                     | codegen kind                        |
|--------|------------------------------|-------------------------------------|
| cpu    | matmul `a@b`                 | extern kernel (no Inductor codegen) |
| cpu    | pointwise `(a*b+c).relu()`   | C++ / OpenMP                        |
| cpu    | reduction `softmax(a,-1)`    | C++ / OpenMP                        |
| mps    | matmul `a@b`                 | extern kernel (no Inductor codegen) |
| mps    | pointwise `(a*b+c).relu()`   | Metal / MPS                         |
| mps    | reduction `softmax(a,-1)`    | Metal / MPS                         |

On this machine the two GPU-class backends diverge exactly as their kernel
languages would suggest. Where Inductor generates a kernel, CPU gets vectorized,
OpenMP-parallel C++ (`cpp_fused_...`, `at::vec::`, `#pragma omp`), and MPS gets a
Metal shader (`async_compile.metal(...)`, `kernel void ...`, threadgroup buffers
for the softmax reduction). Neither device emits Triton, because Triton is
Inductor's GPU path for CUDA and XPU, not for CPU or Apple Metal.

The matmul is a separate case on both devices. Inductor generates no kernel of
its own and dispatches to an external library op through `extern_kernels.mm(...)`,
which is BLAS on CPU and the MPS matmul on MPS. The same shape holds on CUDA,
where such ops go to cuBLAS.

### Full output

```
======================================================================
ENVIRONMENT
======================================================================
torch.__version__ ............. 2.12.1
default torch.compile backend . 'inductor'
cuda available ................ False
mps available ................. True
xpu available ................. False
devices probed:
  - cpu   arm
  - mps   Apple GPU (Metal / MPS)

----------------------------------------------------------------------
[cpu] matmul (a@b)
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
[cpu] reduction softmax(a,-1)
----------------------------------------------------------------------
codegen kind : C++ / OpenMP
captured via : run_and_get_triton_code
generated kernel (excerpt):
(from line 32)
cpp_fused__softmax_0 = async_compile.cpp_pybinding(['const float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(6)
    {
        int tid = omp_get_thread_num();
        {
            std::unique_ptr<float []> buf_local_buffer_data_0 = std::make_unique<float []>(128LL);
            float* local_buffer_data_0 = buf_local_buffer_data_0.get();
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(64LL); x0+=static_cast<int64_t>(1LL))
... (more lines)

----------------------------------------------------------------------
[mps] matmul (a@b)
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

----------------------------------------------------------------------
[mps] reduction softmax(a,-1)
----------------------------------------------------------------------
codegen kind : Metal / MPS
captured via : run_and_get_triton_code
generated kernel (excerpt):
(from line 31)
generated_kernel_0 = async_compile.metal('generated_kernel_0', '''

    [[max_total_threads_per_threadgroup(128)]]
    kernel void generated_kernel_0(
        device float* out_ptr2,
        constant float* in_ptr0,
        uint2 thread_pos [[thread_position_in_grid]],
        uint2 group_pos [[thread_position_in_threadgroup]]
    ) {
        auto xindex = thread_pos.x;
        auto r0_index = thread_pos.y;
        int r0_1 = r0_index;
        int x0 = xindex;
        threadgroup float tmp_acc_0[4];
        threadgroup float tmp_acc_1[4];
... (more lines)

======================================================================
SUMMARY:  device x function  ->  codegen kind
======================================================================
  cpu  |  matmul (a@b)              ->  extern kernel (no Inductor codegen)   (run_and_get_triton_code)
  cpu  |  pointwise (a*b+c).relu()  ->  C++ / OpenMP   (run_and_get_triton_code)
  cpu  |  reduction softmax(a,-1)   ->  C++ / OpenMP   (run_and_get_triton_code)
  mps  |  matmul (a@b)              ->  extern kernel (no Inductor codegen)   (run_and_get_triton_code)
  mps  |  pointwise (a*b+c).relu()  ->  Metal / MPS   (run_and_get_triton_code)
  mps  |  reduction softmax(a,-1)   ->  Metal / MPS   (run_and_get_triton_code)

======================================================================
FINDINGS  (per device, from the captured output above)
======================================================================

[cpu] arm
  Inductor-generated kernels : C++ / OpenMP
  Triton emitted here        : no
  handed to an extern library : matmul (a@b)

[mps] Apple GPU (Metal / MPS)
  Inductor-generated kernels : Metal / MPS
  Triton emitted here        : no
  handed to an extern library : matmul (a@b)

======================================================================
READING
======================================================================
Inductor is one compiler front-end with several code generators behind it.
The device you place tensors on selects the generator: Triton for CUDA and
XPU GPUs, Metal for Apple MPS, vectorized C++ for CPU. Library-shaped ops
such as matmul are commonly dispatched to an external BLAS-style kernel
instead of being generated at all. The table above shows which path each
backend actually took on this machine, rather than assuming any of them.
```
