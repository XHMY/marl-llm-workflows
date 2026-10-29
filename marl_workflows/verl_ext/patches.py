"""The only module that alters upstream (verl@v0.6.1) behaviour at runtime.

Nothing here runs on import. Each alteration is applied explicitly by the code that needs it:

- P1 :func:`scoped_multi_adapter_peft`, a context manager entered by
  ``MultiAgentAsyncActorRolloutRefWorker._build_model_optimizer`` (IP actor only);
- P2 :func:`apply_worker_patches`, called first thing in ``MultiAgentAsyncActorRolloutRefWorker.init_model``
  (inside each Ray worker process);
- H1 :func:`apply_driver_patches`, called by the task runner before the trainer is built (the process
  that constructs verl's ``AgentLoopManager``).

:data:`PATCHES` documents each one; ``docs/upstream-patches.md`` is rendered from it
(:func:`render_markdown`).

Deliberately **not** patched, and the mechanism used instead:

========================================  ==============================================
Not patched                               Instead
========================================  ==============================================
``rllm.utils.tracking.Tracking``          subclass ``ResumableTracking``
``OpenAIEngine`` request timeout          client proxy in ``LoRAOpenAIEngine``
``AsyncLLMServerManager``                 subclass ``MultiLoRAServerManager``
``rllm VerlEngine``                       subclass ``MultiLoRAVerlEngine``
``DataParallelPPOActor``                  subclass ``MultiAgentDPActor``, re-instantiated in ``init_model``
``vLLMHttpServerBase`` / ``vLLMReplica``  subclasses, selected through H1
``RLHFDataset``                           ``data.custom_cls``
Dataset registry location                 subclass ``MarlDatasetRegistry``
========================================  ==============================================
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "PATCHES",
    "PatchInfo",
    "applied_patches",
    "apply_driver_patches",
    "apply_worker_patches",
    "render_markdown",
    "revert_driver_patches",
    "revert_worker_patches",
    "scoped_multi_adapter_peft",
]

VERL_PIN = "v0.6.1"


@dataclass(frozen=True)
class PatchInfo:
    """One runtime alteration of upstream code.

    ``anchors`` are ``(path inside third_party/verl, 1-based line, text expected on that line)``; the
    tests check them against the pinned submodule so that a re-pin cannot silently move a patch target.
    """

    pid: str
    kind: Literal["patch", "registry"]
    upstream_symbol: str
    upstream_location: str
    reason_no_hook: str
    behaviour_change: str
    scope: Literal["scoped", "worker", "driver"]
    applied_by: str
    anchors: tuple[tuple[str, int, str], ...]


PATCHES: tuple[PatchInfo, ...] = (
    PatchInfo(
        pid="P1",
        kind="patch",
        upstream_symbol="verl.workers.fsdp_workers.get_peft_model (module global imported from peft)",
        upstream_location=(
            f"verl@{VERL_PIN}:verl/workers/fsdp_workers.py:33 (import), :439 (call inside "
            "ActorRolloutRefWorker._build_model_optimizer, :269-577)"
        ),
        reason_no_hook=(
            "The per-role adapters must be added after the stock PEFT wrap and before get_fsdp_wrap_policy / "
            "FSDP wrapping, inside one ~300-line method that takes no callback; FSDP1 flat parameters cannot "
            "be extended after wrapping."
        ),
        behaviour_change=(
            "Only inside the `with` block: get_peft_model(model, cfg) returns the stock PeftModel after "
            "calling hook(peft_model, pristine_copy_of_cfg), which adds one adapter per role and sets "
            "requires_grad. The original is restored in `finally`, also on exceptions. Outside the block "
            "nothing changes."
        ),
        scope="scoped",
        applied_by="MultiAgentAsyncActorRolloutRefWorker._build_model_optimizer (role='actor', IP only)",
        anchors=(
            ("verl/workers/fsdp_workers.py", 33, "get_peft_model"),
            ("verl/workers/fsdp_workers.py", 439, "get_peft_model(actor_module"),
        ),
    ),
    PatchInfo(
        pid="P2",
        kind="registry",
        upstream_symbol='verl.workers.rollout.base._ROLLOUT_REGISTRY[("vllm", "async")]',
        upstream_location=(
            f"verl@{VERL_PIN}:verl/workers/rollout/base.py:81-86 (dict), :89-104 (get_rollout_class), "
            "read at verl/workers/fsdp_workers.py:620"
        ),
        reason_no_hook=(
            "v0.6.1 exposes no register function for rollout classes; the private dict is the only "
            "selection point used by the hybrid worker."
        ),
        behaviour_change=(
            "The hybrid worker builds MultiLoRAvLLMAsyncRollout instead of vLLMAsyncRollout: max_loras is "
            "read from rollout.engine_kwargs.vllm.max_loras instead of being fixed to 1, and update_weights "
            "registers an adapter under the lora_int_id it is given instead of always 123."
        ),
        scope="worker",
        applied_by="MultiAgentAsyncActorRolloutRefWorker.init_model (first line)",
        anchors=(
            ("verl/workers/rollout/base.py", 81, "_ROLLOUT_REGISTRY = {"),
            ("verl/workers/rollout/base.py", 83, '("vllm", "async")'),
            ("verl/workers/fsdp_workers.py", 620, "get_rollout_class("),
        ),
    ),
    PatchInfo(
        pid="H1",
        kind="registry",
        upstream_symbol='verl.workers.rollout.replica.RolloutReplicaRegistry.register("vllm", loader)',
        upstream_location=(
            f"verl@{VERL_PIN}:verl/workers/rollout/replica.py:193-205 (API), :240 (built-in registration); "
            "resolved at verl/experimental/agent_loop/agent_loop.py:682"
        ),
        reason_no_hook="Official API; listed so that every runtime alteration is in one place.",
        behaviour_change=(
            "AgentLoopManager builds MultiLoRAvLLMReplica, whose servers (MultiLoRAvLLMHttpServer) start vLLM "
            "with max_loras from the config and serve each request with the LoRA id it carries."
        ),
        scope="driver",
        applied_by="the task runner (MultiAgentTaskRunner.run) before the trainer is constructed",
        anchors=(
            ("verl/workers/rollout/replica.py", 193, "class RolloutReplicaRegistry"),
            ("verl/workers/rollout/replica.py", 240, 'RolloutReplicaRegistry.register("vllm"'),
            ("verl/experimental/agent_loop/agent_loop.py", 682, "get_rollout_replica_class("),
        ),
    ),
)

# pid -> value replaced by the patch (restored by the revert functions).
_applied: dict[str, Any] = {}

_P2_KEY = ("vllm", "async")
_P2_TARGET = "marl_workflows.verl_ext.vllm_rollout.MultiLoRAvLLMAsyncRollout"


def applied_patches() -> list[str]:
    """Ids of the patches currently in force in this process, in P1, P2, H1 order."""
    return [p.pid for p in PATCHES if p.pid in _applied]


@contextmanager
def scoped_multi_adapter_peft(hook: Callable[[Any, Any], None]) -> Iterator[None]:
    """P1: inside the block, ``fsdp_workers.get_peft_model`` also runs ``hook(peft_model, lora_config)``.

    ``hook`` receives a deep copy of the ``LoraConfig`` taken *before* PEFT mutates it, which equals the
    research fork's fresh ``LoraConfig(**lora_config)`` per extra adapter. Not re-entrant.
    """
    import verl.workers.fsdp_workers as fsdp_workers

    if "P1" in _applied:
        raise RuntimeError("scoped_multi_adapter_peft is not re-entrant")
    original = fsdp_workers.get_peft_model

    def get_peft_model_with_role_adapters(model: Any, peft_config: Any, *args: Any, **kwargs: Any) -> Any:
        pristine = copy.deepcopy(peft_config)
        peft_model = original(model, peft_config, *args, **kwargs)
        hook(peft_model, pristine)
        return peft_model

    fsdp_workers.get_peft_model = get_peft_model_with_role_adapters
    _applied["P1"] = original
    try:
        yield
    finally:
        fsdp_workers.get_peft_model = original
        _applied.pop("P1", None)


def apply_worker_patches() -> None:
    """P2: route ``get_rollout_class("vllm", "async")`` to ``MultiLoRAvLLMAsyncRollout``. Idempotent."""
    from verl.workers.rollout import base

    if "P2" in _applied:
        return
    _applied["P2"] = base._ROLLOUT_REGISTRY.get(_P2_KEY)
    base._ROLLOUT_REGISTRY[_P2_KEY] = _P2_TARGET


def revert_worker_patches() -> None:
    """Undo P2 (tests only)."""
    from verl.workers.rollout import base

    if "P2" not in _applied:
        return
    previous = _applied.pop("P2")
    if previous is None:
        base._ROLLOUT_REGISTRY.pop(_P2_KEY, None)
    else:
        base._ROLLOUT_REGISTRY[_P2_KEY] = previous


def _load_multi_lora_replica() -> type:
    # Lazy: vLLM is imported only when AgentLoopManager resolves the replica class.
    from marl_workflows.verl_ext.vllm_server import load_multi_lora_replica

    return load_multi_lora_replica()


def apply_driver_patches() -> None:
    """H1: register ``MultiLoRAvLLMReplica`` as the ``"vllm"`` rollout replica. Idempotent."""
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    if "H1" in _applied:
        return
    _applied["H1"] = RolloutReplicaRegistry._registry.get("vllm")
    RolloutReplicaRegistry.register("vllm", _load_multi_lora_replica)


def revert_driver_patches() -> None:
    """Undo H1 (tests only)."""
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    if "H1" not in _applied:
        return
    previous = _applied.pop("H1")
    if previous is None:
        RolloutReplicaRegistry._registry.pop("vllm", None)
    else:
        RolloutReplicaRegistry.register("vllm", previous)


def render_markdown() -> str:
    """The patch table of ``docs/upstream-patches.md``."""
    lines = [
        "| id | kind | Upstream symbol | Location | Why no hook | Behavioural change | Scope / applied by |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in PATCHES:
        cells = [
            p.pid,
            p.kind,
            f"`{p.upstream_symbol}`",
            p.upstream_location,
            p.reason_no_hook,
            p.behaviour_change,
            f"{p.scope}; {p.applied_by}",
        ]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(lines) + "\n"
