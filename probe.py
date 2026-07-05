"""
probe.py: what does torch.compile's Inductor backend actually emit, per device?

Empirical question: when we run torch.compile (default backend = Inductor),
does Inductor emit Triton, and does the answer differ between CPU and MPS?

This script does not assume an answer. It compiles two trivial functions,
runs them on each available device to trigger compilation, captures the code
Inductor generated, and classifies it. Read the output, not the comments, to
decide what MPS does.

Run:  uv run python probe.py   (or: python probe.py inside the venv)
Only torch is required.
"""

from __future__ import annotations

import inspect
import io
import logging
import re
import traceback

import torch


# --------------------------------------------------------------------------- #
# The two functions under test: a matmul and a pointwise fusion.
# Kept trivial on purpose. We care about which codegen path is taken, not perf.
# --------------------------------------------------------------------------- #
def matmul(a, b):
    return a @ b


def pointwise(a, b, c):
    # A classic fusible chain: multiply, add, relu.
    return (a * b + c).relu()


FUNCS = {
    "matmul": {
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
#   Triton        : Inductor generated a Triton kernel (@triton.jit / tl.*)
#   Metal / MPS   : Inductor generated a Metal shader (async_compile.metal /
#                   `kernel void`)
#   C++ / OpenMP  : Inductor generated a C++ kernel (async_compile.cpp /
#                   cpp_fused / at::vec / #pragma omp)
#   extern kernel : Inductor generated no kernel of its own and dispatched to an
#                   external library op via extern_kernels.<op>(...), for example
#                   BLAS mm on CPU or the MPS mm on MPS. No Inductor codegen.
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
# Environment facts.
# --------------------------------------------------------------------------- #
def default_compile_backend() -> str:
    try:
        return inspect.signature(torch.compile).parameters["backend"].default
    except Exception:  # noqa: BLE001
        return "unknown"


def available_devices() -> list[str]:
    devices = ["cpu"]
    try:
        if torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:  # noqa: BLE001
        pass
    return devices


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    print(f"torch.__version__ ............. {torch.__version__}")
    print(f"torch.backends.mps.is_available() {torch.backends.mps.is_available()}")
    print(f"default torch.compile backend . {default_compile_backend()!r}")
    devices = available_devices()
    print(f"devices probed ................ {devices}")
    print()

    # results[(device, func_name)] = (kind, method_or_error, code_or_None)
    results: dict[tuple[str, str], tuple[str, str, str | None]] = {}

    for device in devices:
        for func_name, spec in FUNCS.items():
            header = f"[{device}] {func_name}"
            print("-" * 70)
            print(header)
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
            except Exception as exc:  # noqa: BLE001 (MPS may fail; keep going)
                msg = f"{type(exc).__name__}: {exc}"
                results[(device, func_name)] = ("ERROR", msg, None)
                print(f"codegen kind : ERROR")
                print(f"detail       : {msg}")
                print(traceback.format_exc())
            print()

    # ----------------------------------------------------------------------- #
    # Summary table.
    # ----------------------------------------------------------------------- #
    print("=" * 70)
    print("SUMMARY:  device x function  ->  codegen kind")
    print("=" * 70)
    dev_w = max(len(d) for d in devices)
    fn_w = max(len(f) for f in FUNCS)
    for device in devices:
        for func_name in FUNCS:
            kind, method, _ = results[(device, func_name)]
            print(f"  {device:<{dev_w}}  |  {func_name:<{fn_w}}  ->  {kind}   ({method})")
    print()

    # ----------------------------------------------------------------------- #
    # Verdict: is MPS a Triton codegen target, or something else?
    # Derived from the actual captured output, not assumptions.
    # ----------------------------------------------------------------------- #
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    mps_kinds = {
        func_name: results[("mps", func_name)][0]
        for func_name in FUNCS
        if ("mps", func_name) in results
    }
    cpu_kinds = {
        func_name: results[("cpu", func_name)][0]
        for func_name in FUNCS
        if ("cpu", func_name) in results
    }
    print(f"CPU codegen : {cpu_kinds}")
    if not mps_kinds:
        print("MPS was not available on this machine, so there is no MPS verdict.")
        return
    print(f"MPS codegen : {mps_kinds}")

    non_codegen = {"extern kernel (no Inductor codegen)", "fallback / no codegen"}
    generated = {k for k in mps_kinds.values() if k not in non_codegen}
    dispatched = {k for k in mps_kinds.values() if k in non_codegen}

    if not generated:
        print("\n=> On MPS, Inductor generated no kernels here; everything dispatched to")
        print("   external ops. Inconclusive on the Triton question; try more ops.")
    elif generated == {"Triton"}:
        print("\n=> MPS IS a Triton codegen target: Inductor emitted Triton on MPS.")
    elif "Triton" in generated:
        print("\n=> MPS is PARTIALLY a Triton target: some generated kernels are Triton,")
        print("   some are not. See the table above.")
    else:
        path = ", ".join(sorted(generated))
        print(f"\n=> MPS is NOT a Triton codegen target. Where Inductor generated a kernel")
        print(f"   (the fusible pointwise op), it emitted: {path}, not Triton.")
        if dispatched:
            print(f"   Ops with no Inductor codegen (e.g. matmul) went to: {', '.join(sorted(dispatched))}.")


if __name__ == "__main__":
    main()
