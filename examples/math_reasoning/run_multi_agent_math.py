import asyncio
import json
import os
import sys
from copy import deepcopy

from multi_agent_math_workflow import MultiAgentMathWorkflow
from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine import OpenAIEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.rewards.reward_fn import math_reward_fn

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "math_reasoning"))


def load_data(n=1, split="test"):
    """Load DeepMath data using the Dataset interface."""
    dataset = DatasetRegistry.load_dataset("deepmath", split)
    if dataset is None:
        print("Dataset not found, preparing dataset...")
        from prepare_deepmath_data import prepare_deepmath_data

        prepare_deepmath_data()
        dataset = DatasetRegistry.load_dataset("deepmath", split)
        if dataset is None:
            raise ValueError(f"Failed to load {split} dataset after preparation")

    data = []
    for idx, example in enumerate(dataset):
        processed = process_deepmath_fn(example, idx)
        for i in range(n):
            data.append(deepcopy(processed))
    return data


def process_deepmath_fn(example, idx):
    """Process DeepMath example into the expected format."""
    question = example["question"]
    ground_truth_answer = str(example["final_answer"]).strip()
    prompts = example.get("prompts", {})

    task = {
        "question": question,
        "ground_truth_answer": ground_truth_answer,
        "prompts": prompts,
        "idx": idx,
        "data_source": "deepmath"
    }
    return task


def evaluate_results(results):
    """Evaluate the results and compute metrics."""
    from collections import defaultdict

    # Create a map to store correct answers per problem
    problem_correct_map = defaultdict(int)
    problem_total_map = defaultdict(int)

    # Accumulate metrics
    total_generator_acc = 0.0
    total_evaluator_acc = 0.0
    total_refiner_acc = 0.0
    total_iterations = 0.0
    success_count = 0

    # Count correct answers for each problem
    for episode in results:
        problem = episode.task["question"]

        # Use the episode-level is_correct flag set by the workflow
        is_correct = episode.is_correct

        problem_correct_map[problem] += int(is_correct)
        problem_total_map[problem] += 1

        # Accumulate workflow metrics
        metrics = episode.metrics
        total_generator_acc += metrics.get("generator_acc", 0.0)
        total_evaluator_acc += metrics.get("evaluator_acc", 0.0)
        total_refiner_acc += metrics.get("refiner_acc", 0.0)
        total_iterations += metrics.get("total_iterations", 0.0)
        success_count += metrics.get("success", 0)

    # Calculate pass@1 and pass@k
    k = max(problem_total_map.values()) if problem_total_map else 1
    total_problems = len(problem_correct_map)
    total_episodes = len(results)

    if total_problems > 0:
        pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
        pass_at_k = sum(1 for problem, correct in problem_correct_map.items() if correct > 0) / total_problems
    else:
        pass_at_1 = 0.0
        pass_at_k = 0.0

    # Calculate average metrics
    avg_generator_acc = total_generator_acc / total_episodes if total_episodes > 0 else 0.0
    avg_evaluator_acc = total_evaluator_acc / total_episodes if total_episodes > 0 else 0.0
    avg_refiner_acc = total_refiner_acc / total_episodes if total_episodes > 0 else 0.0
    avg_iterations = total_iterations / total_episodes if total_episodes > 0 else 0.0
    success_rate = success_count / total_episodes if total_episodes > 0 else 0.0

    print("\n=== Multi-Agent Math Reasoning Results ===")
    print(f"Total unique problems: {total_problems}")
    print(f"Total episodes: {total_episodes}")
    print(f"\nAccuracy Metrics:")
    print(f"  Pass@1 Accuracy: {pass_at_1:.4f}")
    print(f"  Pass@{k} Accuracy: {pass_at_k:.4f}")
    print(f"\nAgent Performance:")
    print(f"  Generator Accuracy: {avg_generator_acc:.4f}")
    print(f"  Evaluator Accuracy: {avg_evaluator_acc:.4f}")
    print(f"  Refiner Accuracy: {avg_refiner_acc:.4f}")
    print(f"\nWorkflow Metrics:")
    print(f"  Average Iterations: {avg_iterations:.2f}")
    print(f"  Success Rate: {success_rate:.4f}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Configuration
    n_parallel_tasks = 128
    max_refinement_iterations = 3  # Maximum refinement iterations

    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    rollout_engine = OpenAIEngine(
        model=model_name,
        tokenizer=tokenizer,
        max_prompt_length=2048,
        max_response_length=2048,
        base_url="http://localhost:30000/v1",
        api_key="None",
        sampling_params={"temperature": 0.6, "top_p": 0.95},
    )

    engine = AgentWorkflowEngine(
        workflow_cls=MultiAgentMathWorkflow,
        workflow_args={
            "max_refinement_iterations": max_refinement_iterations,
            "reward_function": math_reward_fn,
        },
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )

    # Load DeepMath tasks
    tasks = load_data(n=1, split="test")
    print(f"Loaded {len(tasks)} DeepMath tasks")

    results = asyncio.run(engine.execute_tasks(tasks))

    # Evaluate results (rewards are already assigned in the workflow)
    print("\nEvaluating results...")
    evaluate_results(results)

    # Save results
    os.makedirs("logs", exist_ok=True)
    output_file = "logs/multi_agent_math_results.json"
    with open(output_file, "w") as f:
        json.dump([episode.to_dict() for episode in results], f, indent=4)

    print(f"\nResults saved to {output_file}")
