from datasets import concatenate_datasets, load_dataset
from rllm.data.dataset import DatasetRegistry


def prepare_math_data():
    aime_i = load_dataset("MathArena/aime_2024_I", split="train")
    aime_ii = load_dataset("MathArena/aime_2024_II", split="train")
    test_dataset = concatenate_datasets([aime_i, aime_ii])

    def preprocess_fn(example, idx):
        return {
            "question": example["problem"],
            "final_answer": example["answer"],
            "data_source": "math",
        }

    test_dataset = test_dataset.map(preprocess_fn, with_indices=True)

    test_dataset = DatasetRegistry.register_dataset("aime2024", test_dataset, "test")
    return test_dataset


if __name__ == "__main__":
    test_dataset = prepare_math_data()
    print(f"Registered aime2024 with {len(test_dataset.get_data())} problems")
    print(test_dataset.get_data_path())
