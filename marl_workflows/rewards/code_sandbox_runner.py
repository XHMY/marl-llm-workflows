"""Sandbox runner: grades one code sample in a fresh interpreter (stdlib only).

Executed by path as ``python -I code_sandbox_runner.py`` from ``code_sandbox.run_fresh``; never
imported through the package. Protocol: one JSON request on stdin (``sample``, ``generation``,
``timeout``, ``debug``, ``max_memory_bytes``, ``max_file_bytes``, ``repo_root``, ``sys_path``),
one JSON response ``{"res": [...], "metadata": {...}}`` on the original stdout.

Order of operations: read the request, set the resource limits, close inherited file
descriptors, import upstream's LiveCodeBench harness without executing ``rllm/__init__.py``
(which imports torch) plus numpy and pandas, then call ``run_test``. The harness and the solution see stdin, stdout
and stderr redirected to ``/dev/null``; the response goes to a private duplicate of stdout.
"""

import importlib
import json
import os
import resource
import sys
import types
from importlib.machinery import PathFinder
from pathlib import Path

_PACKAGES = (("rllm", ()), ("rllm.rewards", ("rewards",)), ("rllm.rewards.code_utils", ("rewards", "code_utils")))


def _set_limits(request):
    mem = request.get("max_memory_bytes")
    if mem:
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_DATA, (mem, mem))
    # Kernel-level backup for signal.alarm, which cannot interrupt C-level work holding the
    # GIL; same formula as the global wall-clock cap. Set before run_test() calls
    # reliability_guard(), which disables the resource module.
    try:
        num_tests = len(json.loads(request["sample"]["input_output"]).get("inputs", []))
    except (json.JSONDecodeError, KeyError, TypeError):
        num_tests = 1
    cpu_soft = min((request["timeout"] + 1) * num_tests + 5, 120)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft, cpu_soft + 3))
    file_bytes = request.get("max_file_bytes")
    if file_bytes:
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _rllm_dir(request):
    """Directory of the ``rllm`` package: the pinned submodule, else an installed rllm."""
    roots = [request.get("repo_root"), str(Path(__file__).resolve().parents[2])]
    for root in filter(None, roots):
        candidate = Path(root) / "third_party" / "rllm" / "rllm"
        if (candidate / "rewards" / "code_utils" / "livecodebench.py").is_file():
            return candidate
    spec = PathFinder.find_spec("rllm", request.get("sys_path") or [])
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("cannot locate the rllm package for the LiveCodeBench harness")
    return Path(list(spec.submodule_search_locations)[0])


def _import_run_test(request):
    """Import ``rllm.rewards.code_utils.livecodebench`` through stub parent packages."""
    rllm_dir = _rllm_dir(request)
    for name, parts in _PACKAGES:
        module = types.ModuleType(name)
        module.__path__ = [str(rllm_dir.joinpath(*parts))]
        sys.modules[name] = module
    from rllm.rewards.code_utils.livecodebench import run_test

    # Every solution is prefixed with the harness's BASE_IMPORTS (numpy, pandas). Importing them
    # after reliability_guard() fails (numpy writes os.environ, whose putenv the guard removes),
    # so load them first, as the rllm-importing parent of the fork's forked sandbox had.
    for name in ("numpy", "pandas"):
        try:
            importlib.import_module(name)
        except ImportError:
            pass
    return run_test


def _vsz_mib():
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[0])
    return round(pages * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)


def _plain(value):
    """Per-test result as the parent's ``result[j] == True`` would read it, JSON-safe."""
    if type(value) in (bool, int):
        return value
    try:
        return bool(value == True)  # noqa: E712 (numpy bools and other comparison results)
    except Exception:
        return False


def main():
    request = json.loads(sys.stdin.buffer.read())
    out_fd = os.dup(1)  # non-inheritable: children of the solution never hold the result pipe
    os.closerange(3, out_fd)
    os.closerange(out_fd + 1, 4096)
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    payload = None  # a BaseException (e.g. SystemExit from the solution) sends nothing, as before
    try:
        _set_limits(request)
        run_test = _import_run_test(request)
        sandbox = {"sandbox_vsz_mib": _vsz_mib(), "sandbox_torch_loaded": "torch" in sys.modules}
        res, metadata = run_test(
            request["sample"], test=request["generation"], debug=request.get("debug", False), timeout=request["timeout"]
        )
        res = [_plain(v) for v in res] if isinstance(res, list) else res
        metadata = {**metadata, **sandbox} if isinstance(metadata, dict) else metadata
        payload = json.dumps({"res": res, "metadata": metadata}, default=repr)
    except Exception as e:
        # MemoryError is a subclass of Exception, so this also covers RLIMIT_AS failures.
        payload = json.dumps({"res": [-4], "metadata": {"error_code": -4, "error_message": f"{type(e).__name__}: {e}"}})
    except BaseException:
        pass
    try:
        data = payload.encode() if payload else b""
        while data:
            data = data[os.write(out_fd, data) :]
    finally:
        # Skip interpreter teardown (atexit handlers may fail under RLIMIT_AS).
        os._exit(0)


if __name__ == "__main__":
    main()
