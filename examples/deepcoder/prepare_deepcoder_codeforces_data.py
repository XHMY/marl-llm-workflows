import json

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry
from rllm.data.utils import fetch_live_code_bench_system_prompt


def prepare_deepcoder_codeforces_data():
    dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="codeforces", split="test")

    def preprocess_fn(example, idx):
        starter_code = example.get("starter_code", "")
        question = fetch_live_code_bench_system_prompt(example["problem"], starter_code if starter_code else None)

        tests_raw = example["tests"]
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

        if not isinstance(tests, list):
            tests = [tests] if tests else []

        for test in tests:
            if test.get("testtype") == "functional" and metadata.get("func_name") is not None:
                test["metadata"] = {"func_name": str(metadata["func_name"])}
            else:
                test["metadata"] = {"func_name": None}

        return {
            "question": question,
            "ground_truth": json.dumps(tests),
            "data_source": "livecodebench",
            "uid": f"codeforces_{idx}",
            "index": idx,
            "starter_code": starter_code or "",
            "metadata": json.dumps(metadata if isinstance(metadata, dict) else {}),
        }

    dataset = dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=1, remove_columns=dataset.column_names)
    dataset = DatasetRegistry.register_dataset("deepcoder_codeforces", dataset, "test")
    return dataset


if __name__ == "__main__":
    dataset = prepare_deepcoder_codeforces_data()
    print(f"Registered deepcoder_codeforces with {len(dataset.get_data())} problems")
    print(dataset.get_data_path())
