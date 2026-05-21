"""Test code_reward_fn inside ProcessPoolExecutor to reproduce eval 0% accuracy."""
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from rllm.rewards.code_reward import set_direct_execution


def _code_executor_init():
    set_direct_execution(True)


def test_reward_direct(task, code):
    """Run code_reward_fn with _USE_DIRECT_EXECUTION=True (like eval)."""
    from rllm.rewards.reward_fn import code_reward_fn
    result = code_reward_fn(task, code)
    return {'reward': result.reward, 'is_correct': result.is_correct, 'metadata': result.metadata}


def test_reward_v2(task, code):
    """Run code_reward_fn with _USE_DIRECT_EXECUTION=False (default)."""
    from rllm.rewards.reward_fn import code_reward_fn
    result = code_reward_fn(task, code)
    return {'reward': result.reward, 'is_correct': result.is_correct, 'metadata': result.metadata}


if __name__ == "__main__":
    traj_path = "checkpoints/rllm-workflow-MARL-v2/single_agent-qwen3_0.6b-multi_lora-deepcoder_primeintellect/evaluation_trajectories/step_10/eval_0.json"
    with open(traj_path) as f:
        d = json.load(f)
    task = d['task']
    action = d['trajectories'][0]['steps'][0]['action']

    print("=== Test 1: Direct call (no executor, _USE_DIRECT_EXECUTION=False) ===")
    r1 = test_reward_v2(task, action)
    print(f"  is_correct: {r1['is_correct']}, reward: {r1['reward']}")
    print(f"  metadata: {json.dumps(r1['metadata'], indent=2)[:500]}")

    print("\n=== Test 2: Direct call with _USE_DIRECT_EXECUTION=True ===")
    set_direct_execution(True)
    r2 = test_reward_direct(task, action)
    print(f"  is_correct: {r2['is_correct']}, reward: {r2['reward']}")
    print(f"  metadata: {json.dumps(r2['metadata'], indent=2)[:500]}")
    set_direct_execution(False)  # reset

    print("\n=== Test 3: Inside ProcessPoolExecutor (spawn, with _code_executor_init) ===")
    mp_ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=mp_ctx, initializer=_code_executor_init) as executor:
        future = executor.submit(test_reward_direct, task, action)
        r3 = future.result(timeout=60)
        print(f"  is_correct: {r3['is_correct']}, reward: {r3['reward']}")
        print(f"  metadata: {json.dumps(r3['metadata'], indent=2)[:500]}")

    print("\n=== Test 4: Inside ProcessPoolExecutor (spawn, WITHOUT _code_executor_init) ===")
    with ProcessPoolExecutor(max_workers=2, mp_context=mp_ctx) as executor:
        future = executor.submit(test_reward_v2, task, action)
        r4 = future.result(timeout=60)
        print(f"  is_correct: {r4['is_correct']}, reward: {r4['reward']}")
        print(f"  metadata: {json.dumps(r4['metadata'], indent=2)[:500]}")

    print("\nDone!")
