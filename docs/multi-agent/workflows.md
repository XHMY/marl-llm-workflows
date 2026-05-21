# Workflows

This page describes the three multi-agent workflow topologies the paper
trains, plus the single-agent baseline used as a control. Each workflow is a
subclass of `rllm.workflows.workflow.Workflow` and produces an `Episode`
containing one `Trajectory` per role invocation.

## Trajectory ID convention

All multi-agent workflows tag rollouts with a `{uid}_{agent_name}` trajectory
ID. The trainer extracts the role with `traj_id.rsplit("_", 1)[1]` to route
each rollout to the correct LoRA adapter (under isolated-policy training).
Agent names must not contain underscores; use names like `generator`,
`evaluator`, `aggregator`, `orchestrator`, `worker`, `synthesizer`.

Numeric suffixes (`generator0`, `generator1`) are stripped by
`_split_batch_by_agent()` in the trainer so that N same-role agents share one
adapter.

## Voting

The voting workflow runs the same task through N parallel generators and then
asks an aggregator to pick the best answer.

```
Generator (parallel × N) ─┐
                          ├─→ Aggregator ─→ Final Answer
Generator (parallel × N) ─┘
```

- **Generator**: produces N independent responses for diverse perspectives.
- **Aggregator**: reviews all N responses and emits a single selection.

### Files

| File | Purpose |
|---|---|
| `rllm/workflows/voting_workflow.py` | Abstract base class (domain-agnostic). |
| `rllm/workflows/voting_v2_workflow.py` | v2 variant with per-agent rewards (no outcome broadcast). |
| `examples/math_reasoning/voting_math_workflow.py` | Math-specific concrete subclass. |
| `examples/math_reasoning/train_voting_math.py` | Training entry point. |

### Abstract methods

Implement these in a subclass:

- `build_generator_prompt(task)` — generation prompt.
- `build_aggregator_prompt(task, responses)` — aggregation prompt with all responses.
- `parse_aggregator_response(response, candidates)` — parse selection from aggregator output.
- `compute_generator_reward(task, response)` — per-generator reward.
- `compute_aggregator_reward(task, selected_response)` — aggregator reward.

### Parallel execution

```python
async def run(self, task, uid, **kwargs):
    generation_tasks = [
        self._generate_single(task, i)
        for i in range(self.n_votes)
    ]
    generator_trajectories = await asyncio.gather(*generation_tasks)

    agg_output = await self.rollout_engine.get_model_response(
        agg_messages,
        agent_name=self.AGGREGATOR_NAME,
    )
```

### Config

```yaml
trainer:
  agent_names: ['generator', 'aggregator']

+rllm.workflow.n_votes: 3
```

### Episode shape

With `n_votes=3`, each episode contains:

```
[generator(vote1), generator(vote2), generator(vote3), aggregator]
```

### Metrics

- `generator_acc` — mean correctness across all N generators.
- `aggregator_acc` — correctness of the aggregator's selection.
- `n_votes` — number of parallel generations.
- `any_correct` — whether any generator was correct (pass@N oracle).
- `success` — final episode correctness (0 or 1).

## Evaluator-Optimizer

The evaluator-optimizer workflow iteratively refines a single response based
on critic feedback.

```
Generator (initial) ─→ Evaluator ─→ Generator (refine) ─→ Evaluator ─→ …
```

The loop continues until the evaluator emits an "is_satisfied" verdict or
`max_iterations` is reached. The same `generator` role handles both initial
generation and refinement.

### Files

| File | Purpose |
|---|---|
| `rllm/workflows/evaluator_optimizer_workflow.py` | Abstract base class. |
| `rllm/workflows/evaluator_optimizer_v2_workflow.py` | v2 variant with per-agent rewards. |
| `examples/math_reasoning/evaluator_optimizer_math_workflow.py` | Math-specific subclass. |
| `examples/math_reasoning/train_evaluator_optimizer_math.py` | Training entry point. |

### Abstract methods

- `build_generator_prompt(task)` — initial generation prompt.
- `build_evaluator_prompt(task, response, iteration, history)` — evaluation prompt.
- `build_refinement_prompt(task, response, evaluation, iteration, history)` — refinement prompt.
- `parse_evaluation(response)` — parse evaluator response into an `EvaluationResult`.
- `compute_generator_reward(task, response)` — generator reward.
- `compute_evaluator_reward(task, response, evaluation, ground_truth_correct)` — evaluator reward.

### EvaluationResult

```python
@dataclass
class EvaluationResult:
    is_satisfied: bool      # controls loop termination
    feedback: str           # feedback string fed back to the generator
    verdict: str            # e.g., "correct", "incorrect"
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
```

### Config

```yaml
trainer:
  agent_names: ['generator', 'evaluator']

+rllm.workflow.max_iterations: 3
```

## Orchestrator-Workers

The orchestrator-workers workflow decomposes a task into multiple solution
strategies, executes each one in parallel, and synthesizes the final answer.

```
Orchestrator (propose strategies) ─→ Workers (parallel × N) ─→ Synthesizer ─→ Final Answer
```

- **Orchestrator**: proposes N distinct strategies for the task.
- **Workers**: each worker solves the full task using one assigned strategy.
- **Synthesizer**: compares worker solutions and picks (or composes) the final answer.

### Files

| File | Purpose |
|---|---|
| `rllm/workflows/orchestrator_workers_workflow.py` | Abstract base class. |
| `examples/math_reasoning/orchestrator_workers_math_workflow.py` | Math-specific subclass. |
| `examples/math_reasoning/train_orchestrator_workers_math.py` | Training entry point. |

### Episode shape

With 3 workers, each episode contains:

```
[orchestrator, worker(strategy1), worker(strategy2), worker(strategy3), synthesizer]
```

The same role name `worker` is used for all N worker rollouts; the trainer's
batch-split routes them to a single shared adapter (under IP) so they form
the per-episode-frequency-asymmetry case studied in the paper.

### Config

```yaml
trainer:
  agent_names: ['orchestrator', 'worker', 'synthesizer']

+rllm.workflow.n_workers: 3
```

## Single-Agent Baseline

The single-agent baseline is used as the paper's control configuration: it
removes the multi-agent topology entirely and runs vanilla
single-policy RL on the same task. Use it to compute the *Δ over base model*
attributable to multi-agent training versus single-agent training.

### Files

| File | Purpose |
|---|---|
| `rllm/workflows/env_single_agent_workflow.py` | Generic single-agent wrapper around any `BaseAgent` × `BaseEnv` pair. |
| `examples/math_reasoning/single_agent_math_workflow.py` | Math-specific concrete subclass. |
| `examples/math_reasoning/train_single_agent_math.py` | Training entry point. |

### Config

```yaml
trainer:
  agent_names: ['generator']
  share_policy: true
```

A single-agent run with one adapter is structurally equivalent to standard
single-policy PPO. It is included here because the paper's
`{Workflow}-{Policy}-{Scale}-{Task}` matrix treats the single-agent baseline
as a first-class cell.

## Environment-coupled variants

For tasks where each step needs an environment interaction (browser, game,
tool calls), two environment-coupled workflows wrap arbitrary
`agent_cls` / `env_cls` pairs:

- `EnvSingleAgentWorkflow` — single-agent loop with environment steps.
- `EnvEvaluatorOptimizerWorkflow` — actor proposes an action, an evaluator
  reviews it, the actor refines if rejected, then the approved action is
  executed in the environment.

These are not used in the paper's main experiments (which are pure-LLM
math and code tasks) but ship with the codebase for completeness.

```python
trainer = AgentTrainer(
    workflow_class=EnvEvaluatorOptimizerWorkflow,
    workflow_args={
        "agent_cls": MiniWobAgent,
        "env_cls": BrowserGymEnv,
        "agent_args": {"use_html": True, "use_axtree": True},
        "env_args": {"miniwob_url": url},
        "max_steps": 10,
        "max_refine_iterations": 2,
    },
    config=config,
    ...
)
```
