import json
import os

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry
from rllm.data.utils import fetch_live_code_bench_system_prompt


def prepare_deepcoder_data(train_size: int = None, test_size: int = None, dataset_name: str = "deepcoder_primeintellect"):
    # Load runtime results for test case filtering
    runtime_results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "coding_dataset_filtering", "runtime_results.json"
    )
    with open(runtime_results_path) as f:
        runtime_results = json.load(f)

    # Load only primeintellect subset
    dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="primeintellect", split="train")

    # Filter test cases to only keep those that passed runtime checks
    def filter_tests_by_runtime(example, idx, runtime_results):
        problem_key = str(idx)
        if problem_key not in runtime_results:
            example["_num_passing_tests"] = -1
            return example

        problem_runtime = runtime_results[problem_key]
        tests_raw = example["tests"]
        if isinstance(tests_raw, str):
            tests = json.loads(tests_raw)
        else:
            tests = tests_raw

        filtered_tests = [
            test for test_idx, test in enumerate(tests)
            if str(test_idx) in problem_runtime
            and problem_runtime[str(test_idx)].get("status") == "passed"
        ]

        example["tests"] = json.dumps(filtered_tests)
        example["_num_passing_tests"] = len(filtered_tests)
        return example

    dataset = dataset.map(
        filter_tests_by_runtime, with_indices=True, num_proc=1,
        fn_kwargs={"runtime_results": runtime_results}
    )
    dataset = dataset.filter(lambda example: example["_num_passing_tests"] != 0)
    dataset = dataset.remove_columns(["_num_passing_tests"])

    # Train/test split: 1000 for test, rest for train
    dataset = dataset.shuffle(seed=42)
    test_dataset = dataset.select(range(1000))
    train_dataset = dataset.select(range(1000, len(dataset)))

    def preprocess_fn(example, idx):
        starter_code = example.get("starter_code", "")
        question = fetch_live_code_bench_system_prompt(example["problem"], starter_code if starter_code else None)

        tests_raw = example["tests"]
        # Handle different test formats
        if isinstance(tests_raw, str):
            tests = json.loads(tests_raw)
        else:
            tests = tests_raw
        metadata = example.get("metadata", {})

        # Convert TACO format to standard format
        if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
            normalized_tests = []
            for input_val, output_val in zip(tests["inputs"], tests["outputs"], strict=False):
                normalized_tests.append({"input": input_val, "output": output_val, "testtype": "stdin_stdout"})
            tests = normalized_tests

        # Ensure tests is always a list
        if not isinstance(tests, list):
            tests = [tests] if tests else []

        for test in tests:
            if test.get("testtype") == "functional" and metadata.get("func_name") is not None:
                test["metadata"] = {"func_name": str(metadata["func_name"])}
            else:
                test["metadata"] = {"func_name": None}

        return {"question": question, "ground_truth": json.dumps(tests), "data_source": "livecodebench", "uid": f"deepcoder_{idx}", "index": idx, "starter_code": starter_code, "metadata": json.dumps(metadata)}

    if train_size:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
    if test_size:
        test_dataset = test_dataset.select(range(min(test_size, len(test_dataset))))

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16, remove_columns=test_dataset.column_names)
    train_dataset = DatasetRegistry.register_dataset(dataset_name, train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset(dataset_name, test_dataset, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_deepcoder_data()
    print(f"  - Train dataset: {len(train_dataset.get_data())} examples")
    print(f"  - Test dataset: {len(test_dataset.get_data())} examples")
    print(train_dataset.get_data()[0])
