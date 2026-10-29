# Upstream patches

`marl-llm-workflows` runs on unmodified submodules (`third_party/verl` at tag `v0.6.1`,
`third_party/rllm` at `186d0bbb`). Everything that changes upstream behaviour is a subclass, except the
three runtime alterations below. All of them live in `marl_workflows/verl_ext/patches.py`; none is applied
on import (`patches.applied_patches()` is empty right after `import marl_workflows.verl_ext.workers`).
Each is applied explicitly by the code named in the last column. `tests/test_patches.py` checks that every
anchor line still exists in the pinned verl and that this table equals `patches.render_markdown()`.

| id | kind | Upstream symbol | Location | Why no hook | Behavioural change | Scope / applied by |
|---|---|---|---|---|---|---|
| P1 | patch | `verl.workers.fsdp_workers.get_peft_model (module global imported from peft)` | verl@v0.6.1:verl/workers/fsdp_workers.py:33 (import), :439 (call inside ActorRolloutRefWorker._build_model_optimizer, :269-577) | The per-role adapters must be created in place of the stock PEFT wrap and before get_fsdp_wrap_policy / FSDP wrapping, inside one ~300-line method that takes no callback; FSDP1 flat parameters cannot be extended or pruned after wrapping. | Only inside the `with` block: get_peft_model(model, cfg) returns build(peft.get_peft_model, model, pristine_copy_of_cfg), which creates the first role's adapter as PEFT's initial adapter, adds one adapter per further role and sets requires_grad, so no adapter named 'default' exists. The original is restored in `finally`, also on exceptions. Outside the block nothing changes. | scoped; MultiAgentAsyncActorRolloutRefWorker._build_model_optimizer (role='actor', IP only) |
| P2 | registry | `verl.workers.rollout.base._ROLLOUT_REGISTRY[("vllm", "async")]` | verl@v0.6.1:verl/workers/rollout/base.py:81-86 (dict), :89-104 (get_rollout_class), read at verl/workers/fsdp_workers.py:620 | v0.6.1 exposes no register function for rollout classes; the private dict is the only selection point used by the hybrid worker. | The hybrid worker builds MultiLoRAvLLMAsyncRollout instead of vLLMAsyncRollout: max_loras is read from rollout.engine_kwargs.vllm.max_loras instead of being fixed to 1, and update_weights registers an adapter under the lora_int_id it is given instead of always 123. | worker; MultiAgentAsyncActorRolloutRefWorker.init_model (first line) |
| H1 | registry | `verl.workers.rollout.replica.RolloutReplicaRegistry.register("vllm", loader)` | verl@v0.6.1:verl/workers/rollout/replica.py:193-205 (API), :240 (built-in registration); resolved at verl/experimental/agent_loop/agent_loop.py:682 | Official API; listed so that every runtime alteration is in one place. | AgentLoopManager builds MultiLoRAvLLMReplica, whose servers (MultiLoRAvLLMHttpServer) start vLLM with max_loras from the config and serve each request with the LoRA id it carries. | driver; the task runner (MultiAgentTaskRunner.run) before the trainer is constructed |

Scopes:

- **scoped** (P1): active only inside a `with` block in one method call; restored in `finally`.
- **worker** (P2): applied inside each Ray actor/rollout worker process by `init_model`.
- **driver** (H1): applied in the process that builds verl's `AgentLoopManager` (the task runner).

## Deliberately not patched

| Upstream code | Mechanism used instead |
|---|---|
| `rllm.utils.tracking.Tracking` | subclass `ResumableTracking` |
| `OpenAIEngine` request timeout | client proxy in `LoRAOpenAIEngine` |
| `verl` `AsyncLLMServerManager` | subclass `MultiLoRAServerManager` (`marl_workflows/engine/verl_engine.py`) |
| `rllm` `VerlEngine` | subclass `MultiLoRAVerlEngine` |
| `verl` `ActorRolloutRefWorker` / `AsyncActorRolloutRefWorker` | subclass `MultiAgentAsyncActorRolloutRefWorker` (`verl_ext/workers.py`) |
| `verl` `DataParallelPPOActor` | subclass `MultiAgentDPActor`, re-instantiated in `init_model` |
| `verl` `vLLMAsyncRollout` | subclass `MultiLoRAvLLMAsyncRollout`, selected by P2 |
| `verl` `vLLMHttpServerBase` / `vLLMReplica` | subclasses `MultiLoRAvLLMHttpServer(Base)` / `MultiLoRAvLLMReplica`, selected by H1 |
| `verl.utils.fsdp_utils.collect_lora_params` / `layered_summon_lora_params` | marked copies with an `adapter_name` argument (`verl_ext/lora_utils.py`) |
| `RLHFDataset` | `data.custom_cls` |
| Dataset registry location | subclass `MarlDatasetRegistry` |

## Research-fork changes that were not carried over

Gradient probes, the coherence regularizer, MD5 weight hashing in the sync path, the sync-mode (SPMD)
`vLLMRollout` `max_loras` edit (rLLM runs `rollout.mode=async`), the least-active load balancer
(no effect on results), and `AgentLoopManager.load_lora_adapter`.
