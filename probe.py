"""
probe.py: what does torch.compile's Inductor backend actually emit, per device?

Empirical question: when we run torch.compile (default backend = Inductor), what
code does Inductor generate, and how does that change across the compute backends
available on this machine (CPU, CUDA, MPS, XPU)?

This script does not assume an answer. It compiles a few trivial functions, runs
each one on every available device to trigger compilation, captures the code
Inductor generated, and classifies it. Read the output, not the comments, to see
which codegen path each backend takes.

Run:  uv run python probe.py   (or: python probe.py inside the venv)
Only torch is required.
"""

from __future__ import annotations

import inspect
import io
import logging
import platform
import re
import traceback

import torch


# --------------------------------------------------------------------------- #
# The functions under test. Each exercises a different codegen shape:
#   matmul     : a library-level op that Inductor tends to hand off wholesale
#   pointwise  : an elementwise chain that Inductor can fuse into one kernel
#   reduction  : a softmax, which fuses reductions (max, sum) with pointwise work
# Kept trivial on purpose. We care about which codegen path is taken, not perf.
# --------------------------------------------------------------------------- #
def matmul(a, b):
    return a @ b


def pointwise(a, b, c):
    # A classic fusible chain: multiply, add, relu.
    return (a * b + c).relu()


def reduction(a):
    # Softmax over the last dim: max-reduce, subtract, exp, sum-reduce, divide.
    return torch.softmax(a, dim=-1)


FUNCS = {
    "matmul (a@b)": {
        "fn": matmul,
        "make_args": lambda dev: (
            torch.randn(64, 64, device=dev),
            torch.randn(64, 64, device=dev),
        ),
    },
    "pointwise (a*b+c).relu()": {
        "fn": pointwise,
        "make_args": lambda dev: (
            torch.randn(1024, device=dev),
            torch.randn(1024, device=dev),
            torch.randn(1024, device=dev),
        ),
    },
    "reduction softmax(a,-1)": {
        "fn": reduction,
        "make_args": lambda dev: (torch.randn(64, 128, device=dev),),
    },
}


# --------------------------------------------------------------------------- #
# Capturing the generated code.
#
# Primary path: torch._inductor.utils.run_and_get_triton_code. It runs the
# compiled fn and returns the generated output code as a string. Despite the
# name, it returns whatever Inductor emitted for the target (Triton, C++, ...).
#
# The helper's import location and even its existence have moved across torch
# releases, so we degrade gracefully:
#   1. run_and_get_triton_code  (returns one code string)
#   2. run_and_get_code         (returns a list of code strings)
#   3. logging capture via torch._logging.set_logs(output_code=True)
# --------------------------------------------------------------------------- #
def _capture_run_and_get_triton_code(compiled, args):
    from torch._inductor.utils import run_and_get_triton_code  # may not exist

    code = run_and_get_triton_code(compiled, *args)
    return code, "run_and_get_triton_code"


def _capture_run_and_get_code(compiled, args):
    from torch._inductor.utils import run_and_get_code  # more general fallback

    _result, code_list = run_and_get_code(compiled, *args)
    return "\n\n# ---- next generated module ----\n\n".join(code_list), "run_and_get_code"


def _capture_via_logs(compiled, args):
    # Last resort: enable the "output_code" logging artifact and capture it by
    # attaching a handler to the torch loggers, then run the compiled fn once.
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)

    root = logging.getLogger("torch")
    prev_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)

    try:
        torch._logging.set_logs(output_code=True)
        compiled(*args)  # triggers compile + logs the output code
    finally:
        try:
            torch._logging.set_logs()  # reset to defaults
        except Exception:
            pass
        root.removeHandler(handler)
        root.setLevel(prev_level)

    text = buf.getvalue()
    if not text.strip():
        raise RuntimeError("output_code logging produced no captured text")
    return text, "TORCH_LOGS=output_code (logging capture)"


def capture_generated_code(compiled, args):
    """Return (code_string, method_label). Raises if every path fails."""
    errors = []
    for capture in (
        _capture_run_and_get_triton_code,
        _capture_run_and_get_code,
        _capture_via_logs,
    ):
        try:
            return capture(compiled, args)
        except Exception as exc:  # noqa: BLE001 (we want to try the next path)
            errors.append(f"{capture.__name__}: {exc!r}")
    raise RuntimeError("all capture paths failed:\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------- #
# Classification of the captured code.
#
# We distinguish, in priority order:
#   Triton        : Inductor generated a Triton kernel (@triton.jit / tl.*).
#                   This is the GPU codegen path used for CUDA and XPU.
#   Metal / MPS   : Inductor generated a Metal shader (async_compile.metal /
#                   `kernel void`). This is the Apple Silicon GPU path.
#   C++ / OpenMP  : Inductor generated a C++ kernel (async_compile.cpp /
#                   cpp_fused / at::vec / #pragma omp). This is the CPU path.
#   extern kernel : Inductor generated no kernel of its own and dispatched to an
#                   external library op via extern_kernels.<op>(...), for example
#                   a BLAS/cuBLAS matmul. No Inductor codegen for that op.
#   fallback      : none of the above matched.
#
# Note: the wrapper module always *imports* `extern_kernels`, so we must look for
# an actual call site (`extern_kernels.mm(`), not the mere import.
# --------------------------------------------------------------------------- #
_EXTERN_CALL_RE = re.compile(r"extern_kernels\.\w+\(")


def classify_codegen(code: str) -> str:
    low = code.lower()

    triton_gen = ("@triton.jit", "async_compile.triton", "tl.load", "tl.store")
    metal_gen = ("async_compile.metal", "kernel void")
    cpp_gen = ("async_compile.cpp", "cpp_fused", "at::vec", "#pragma omp")

    if any(m in code for m in triton_gen):
        return "Triton"
    if any(m in low for m in metal_gen):
        return "Metal / MPS"
    if any(m in low for m in cpp_gen):
        return "C++ / OpenMP"
    if _EXTERN_CALL_RE.search(code):
        return "extern kernel (no Inductor codegen)"
    return "fallback / no codegen"


# Anchors that mark the *interesting* part of the wrapper module, so the excerpt
# shows the real kernel (or the extern dispatch) instead of the import preamble.
_EXCERPT_ANCHORS = (
    "async_compile.triton",
    "@triton.jit",
    "async_compile.metal",
    "kernel void",
    "async_compile.cpp_pybinding",
    "async_compile.cpp",
    "cpp_fused",
)


def interesting_excerpt(code: str, n: int = 15) -> str:
    """First ~n lines of the generated kernel (or extern dispatch), not the
    boilerplate import header that every wrapper module repeats verbatim."""
    lines = code.splitlines()

    start = None
    for idx, line in enumerate(lines):
        if any(a in line for a in _EXCERPT_ANCHORS):
            start = idx
            break
    if start is None:
        # No generated kernel, so anchor on the extern dispatch call in call().
        for idx, line in enumerate(lines):
            if _EXTERN_CALL_RE.search(line):
                start = max(0, idx - 3)  # a little context above the call
                break
    if start is None:
        start = 0

    window = lines[start : start + n]
    prefix = f"(from line {start})\n" if start else ""
    out = prefix + "\n".join(window)
    remaining = len(lines) - (start + n)
    if remaining > 0:
        out += f"\n... ({remaining} more lines)"
    return out


# --------------------------------------------------------------------------- #
# Environment and device discovery.
# --------------------------------------------------------------------------- #
def default_compile_backend() -> str:
    try:
        return inspect.signature(torch.compile).parameters["backend"].default
    except Exception:  # noqa: BLE001
        return "unknown"


def _mps_available() -> bool:
    try:
        return torch.backends.mps.is_available()
    except Exception:  # noqa: BLE001
        return False


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _xpu_available() -> bool:
    try:
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:  # noqa: BLE001
        return False


def discover_devices() -> list[tuple[str, str]]:
    """Return (device_string, human_label) for every backend we can probe.

    CPU is always present. The accelerators are added only when their runtime
    reports itself available, so the same script runs unchanged on a laptop with
    only MPS, a CUDA box, an Intel XPU host, or a plain CPU machine.
    """
    devices: list[tuple[str, str]] = [("cpu", platform.processor() or platform.machine())]

    if _cuda_available():
        try:
            cc = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
            label = f"{torch.cuda.get_device_name(0)} (CUDA, sm_{cc})"
        except Exception:  # noqa: BLE001
            label = "CUDA device"
        devices.append(("cuda", label))

    if _mps_available():
        devices.append(("mps", "Apple GPU (Metal / MPS)"))

    if _xpu_available():
        try:
            label = f"{torch.xpu.get_device_name(0)} (Intel XPU)"
        except Exception:  # noqa: BLE001
            label = "Intel XPU device"
        devices.append(("xpu", label))

    return devices


# The GPU-class codegen backends. Whether a device sits here is the substantive
# result: it says which native kernel language Inductor targets for that GPU.
_CODEGEN_KINDS = {"Triton", "Metal / MPS", "C++ / OpenMP"}


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main() -> None:
    devices = discover_devices()

    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    print(f"torch.__version__ ............. {torch.__version__}")
    print(f"default torch.compile backend . {default_compile_backend()!r}")
    print(f"cuda available ................ {_cuda_available()}")
    print(f"mps available ................. {_mps_available()}")
    print(f"xpu available ................. {_xpu_available()}")
    print("devices probed:")
    for dev, label in devices:
        print(f"  - {dev:<5} {label}")
    print()

    # results[(device, func_name)] = (kind, method_or_error, code_or_None)
    results: dict[tuple[str, str], tuple[str, str, str | None]] = {}

    for device, _label in devices:
        for func_name, spec in FUNCS.items():
            print("-" * 70)
            print(f"[{device}] {func_name}")
            print("-" * 70)
            try:
                args = spec["make_args"](device)
                compiled = torch.compile(spec["fn"])
                code, method = capture_generated_code(compiled, args)
                kind = classify_codegen(code)
                results[(device, func_name)] = (kind, method, code)
                print(f"codegen kind : {kind}")
                print(f"captured via : {method}")
                print("generated kernel (excerpt):")
                print(interesting_excerpt(code))
            except Exception as exc:  # noqa: BLE001 (a backend may fail; keep going)
                msg = f"{type(exc).__name__}: {exc}"
                results[(device, func_name)] = ("ERROR", msg, None)
                print("codegen kind : ERROR")
                print(f"detail       : {msg}")
                print(traceback.format_exc())
            print()

    # ----------------------------------------------------------------------- #
    # Summary table.
    # ----------------------------------------------------------------------- #
    print("=" * 70)
    print("SUMMARY:  device x function  ->  codegen kind")
    print("=" * 70)
    dev_w = max(len(d) for d, _ in devices)
    fn_w = max(len(f) for f in FUNCS)
    for device, _label in devices:
        for func_name in FUNCS:
            kind, method, _ = results[(device, func_name)]
            print(f"  {device:<{dev_w}}  |  {func_name:<{fn_w}}  ->  {kind}   ({method})")
    print()

    # ----------------------------------------------------------------------- #
    # Findings: one neutral, data-driven read per device. No device is treated
    # as the special case; each is reported on the same terms.
    # ----------------------------------------------------------------------- #
    print("=" * 70)
    print("FINDINGS  (per device, from the captured output above)")
    print("=" * 70)
    for device, label in devices:
        kinds = {fn: results[(device, fn)][0] for fn in FUNCS}
        generated = {k for k in kinds.values() if k in _CODEGEN_KINDS}
        dispatched = [fn for fn, k in kinds.items() if "extern kernel" in k]
        errored = [fn for fn, k in kinds.items() if k == "ERROR"]

        print(f"\n[{device}] {label}")
        if generated:
            langs = ", ".join(sorted(generated))
            triton = "yes" if "Triton" in generated else "no"
            print(f"  Inductor-generated kernels : {langs}")
            print(f"  Triton emitted here        : {triton}")
        else:
            print("  Inductor-generated kernels : none captured for these ops")
        if dispatched:
            print(f"  handed to an extern library : {', '.join(dispatched)}")
        if errored:
            print(f"  failed to compile/capture   : {', '.join(errored)}")

    print()
    print("=" * 70)
    print("READING")
    print("=" * 70)
    print(
        "Inductor is one compiler front-end with several code generators behind it.\n"
        "The device you place tensors on selects the generator: Triton for CUDA and\n"
        "XPU GPUs, Metal for Apple MPS, vectorized C++ for CPU. Library-shaped ops\n"
        "such as matmul are commonly dispatched to an external BLAS-style kernel\n"
        "instead of being generated at all. The table above shows which path each\n"
        "backend actually took on this machine, rather than assuming any of them."
    )


if __name__ == "__main__":
    main()
