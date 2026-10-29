"""Launch and manage a LoRA-enabled vLLM OpenAI server for evaluation.

``VLLMServerManager`` is a port of the research fork's ``dashboard/evaluate_checkpoints.py:
601-750`` (same server flags: ``--enable-lora --max-loras --max-lora-rank``), plus startup
registration of adapters (``--lora-modules name=path``) and a command builder that tests can
inspect. Adapters can also be loaded and unloaded at runtime (the evaluation driver does this
per checkpoint); that needs ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=True``, which is set in the
server's environment only, not in this process.

Stand-alone use, serving every role adapter of one checkpoint under its role name::

    python -m marl_workflows.eval.vllm_server --model Qwen/Qwen3-0.6B --port 8000 \\
        --actor-path ckpts/voting-qwen3_0.6b-multi_lora-math/global_step_300/actor \\
        --roles generator aggregator
    python -m marl_workflows.eval.vllm_server --model Qwen/Qwen3-0.6B --lora generator=/path/lora_adapter_generator
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import requests

from marl_workflows.contracts import adapter_dirname

__all__ = ["VLLMServerManager", "lora_modules_for_actor", "main"]

RUNTIME_LORA_ENV = "VLLM_ALLOW_RUNTIME_LORA_UPDATING"


def lora_modules_for_actor(
    actor_path: str | Path,
    agent_names: Sequence[str],
    share_policy: bool,
    prefix: str = "",
) -> dict[str, str]:
    """Served name -> adapter directory for every role of a checkpoint (C14 layout).

    IP: ``{prefix}{role}`` -> ``actor/lora_adapter_{role}``. SP: every ``{prefix}{role}`` ->
    ``actor/lora_adapter`` (one adapter served under each role's name).
    """
    root = Path(actor_path)
    modules: dict[str, str] = {}
    for role in agent_names:
        path = root / adapter_dirname(None if share_policy else role)
        if not path.exists():
            raise FileNotFoundError(f"LoRA adapter for role {role!r} not found at {path}")
        modules[f"{prefix}{role}"] = str(path)
    return modules


class VLLMServerManager:
    """Start / stop a ``vllm.entrypoints.openai.api_server`` subprocess and manage its adapters."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        port: int = 8000,
        max_loras: int = 8,
        max_lora_rank: int = 64,
        max_model_len: int | None = None,
        enable_lora: bool = True,
        lora_modules: Mapping[str, str] | None = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.model = model
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.process: subprocess.Popen | None = None
        self.enable_lora = enable_lora
        self.lora_modules = dict(lora_modules or {})
        # Adapters registered at startup count as loaded, so load_lora skips them.
        self.loaded_loras: set[str] = set()
        self.extra_args = list(extra_args)
        self.config = {
            "tensor_parallel_size": tensor_parallel_size,
            "data_parallel_size": data_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_loras": max_loras,
            "max_lora_rank": max_lora_rank,
            "max_model_len": max_model_len,
        }
        if self.lora_modules and not enable_lora:
            raise ValueError("lora_modules given but enable_lora=False")

    def build_command(self) -> list[str]:
        """The server command line (fork flags, plus ``--lora-modules`` for startup adapters)."""
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model,
            "--tensor-parallel-size",
            str(self.config["tensor_parallel_size"]),
            "--gpu-memory-utilization",
            str(self.config["gpu_memory_utilization"]),
            "--port",
            str(self.port),
            "--trust-remote-code",
        ]
        if self.enable_lora:
            cmd += [
                "--enable-lora",
                "--max-loras",
                str(self.config["max_loras"]),
                "--max-lora-rank",
                str(self.config["max_lora_rank"]),
            ]
            if self.lora_modules:
                cmd += ["--lora-modules", *[f"{name}={path}" for name, path in self.lora_modules.items()]]
        if self.config["data_parallel_size"] > 1:
            cmd += ["--data-parallel-size", str(self.config["data_parallel_size"])]
        if self.config["max_model_len"]:
            cmd += ["--max-model-len", str(self.config["max_model_len"])]
        return cmd + self.extra_args

    def start(self, timeout: int = 600) -> None:
        """Start the server and block until ``/health`` answers."""
        cmd = self.build_command()
        env = dict(os.environ)
        env[RUNTIME_LORA_ENV] = "True"
        print(f"Starting vLLM server: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, stdout=None, stderr=None, text=True, env=env)
        self._wait_for_ready(timeout)
        self.loaded_loras |= set(self.lora_modules)

    def _wait_for_ready(self, timeout: int = 600) -> None:
        print(f"Waiting for vLLM server to be ready (timeout: {timeout}s)...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                if requests.get(f"{self.base_url}/health", timeout=5).status_code == 200:
                    print(f"vLLM server ready at {self.base_url}")
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server process died with exit code: {self.process.returncode}. "
                    "Check the terminal output above for details."
                )
            time.sleep(5)
        raise TimeoutError(f"vLLM server failed to start within {timeout}s")

    def load_lora(self, lora_name: str, lora_path: str) -> None:
        """Register an adapter at runtime under ``lora_name`` (no-op if already loaded)."""
        if not self.enable_lora:
            print(f"Warning: LoRA is disabled, skipping load of {lora_name}")
            return
        if lora_name in self.loaded_loras:
            print(f"LoRA adapter already loaded: {lora_name}")
            return
        response = requests.post(
            f"{self.base_url}/v1/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path},
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to load LoRA {lora_name}: {response.text}")
        self.loaded_loras.add(lora_name)
        print(f"Loaded LoRA adapter: {lora_name} from {lora_path}")

    def unload_lora(self, lora_name: str) -> None:
        """Unregister an adapter (warnings only on failure)."""
        if not self.enable_lora or lora_name not in self.loaded_loras:
            return
        try:
            response = requests.post(
                f"{self.base_url}/v1/unload_lora_adapter", json={"lora_name": lora_name}, timeout=30
            )
            if response.status_code == 200:
                self.loaded_loras.discard(lora_name)
                print(f"Unloaded LoRA adapter: {lora_name}")
            else:
                print(f"Warning: Failed to unload LoRA {lora_name}: {response.text}")
        except Exception as e:  # unloading is best effort between checkpoints
            print(f"Warning: Error unloading LoRA {lora_name}: {e}")

    def unload_all_loras(self) -> None:
        """Unregister every adapter loaded at runtime or startup."""
        for name in list(self.loaded_loras):
            self.unload_lora(name)

    def stop(self) -> None:
        """Terminate the server (kill after 30 s)."""
        if self.process is None:
            return
        print("Stopping vLLM server...")
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process = None
        self.loaded_loras.clear()
        print("vLLM server stopped")


def _parse_lora_pairs(pairs: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        name, sep, path = pair.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--lora expects NAME=PATH, got {pair!r}")
        out[name] = path
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve a base model with role LoRA adapters (vLLM OpenAI API).")
    parser.add_argument("--model", required=True, help="base model id or path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-loras", type=int, default=8)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--data-parallel", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--lora", action="append", default=[], metavar="NAME=PATH", help="adapter to serve (repeat)")
    parser.add_argument("--actor-path", default=None, help="checkpoint actor/ dir; serves each --roles adapter")
    parser.add_argument("--roles", nargs="+", default=None, help="roles to serve from --actor-path")
    parser.add_argument("--share-policy", action="store_true", help="--actor-path holds one shared lora_adapter/")
    parser.add_argument("--no-lora", action="store_true", help="serve the base model only")
    args = parser.parse_args(argv)

    modules = _parse_lora_pairs(args.lora)
    if args.actor_path:
        if not args.roles:
            raise SystemExit("--actor-path needs --roles")
        modules.update(lora_modules_for_actor(args.actor_path, args.roles, args.share_policy))
    server = VLLMServerManager(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        data_parallel_size=args.data_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
        port=args.port,
        max_loras=args.max_loras,
        max_lora_rank=args.max_lora_rank,
        max_model_len=args.max_model_len,
        enable_lora=not args.no_lora,
        lora_modules=modules,
    )

    def _shutdown(_sig: int, _frame: object) -> None:
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    server.start()
    print(f"Serving {sorted(modules) or ['<base model only>']} at {server.base_url}/v1 (Ctrl-C to stop)")
    assert server.process is not None
    server.process.wait()


if __name__ == "__main__":
    main()
