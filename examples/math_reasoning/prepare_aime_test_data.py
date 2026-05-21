from datasets import load_dataset
from rllm.data.dataset import DatasetRegistry


def prepare_math_data():
    test_dataset = load_dataset("MathArena/aime_2025", split="train")

    def preprocess_fn(example, idx):
        return {
            "question": example["problem"],
            "final_answer": example["answer"],
            "data_source": "math",
        }

    test_dataset = test_dataset.map(preprocess_fn, with_indices=True)

    test_dataset = DatasetRegistry.register_dataset("aime2025", test_dataset, "test")
    return test_dataset


if __name__ == "__main__":
    test_dataset = prepare_math_data()
    print(test_dataset.get_data_path())
