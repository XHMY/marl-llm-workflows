"""
Prepare and register DAPO-Math-17k dataset for multi-agent math reasoning training.

This script:
1. Downloads DAPO-Math-17k-Processed from HuggingFace
2. Maps fields to standard format (prompt->question, solution->final_answer)
3. Splits into train/test sets
4. Saves to parquet files in rllm/data/datasets/dapo_math/
5. Registers with DatasetRegistry
"""

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


def prepare_dapo_math_data(
    dataset_name: str = "open-r1/DAPO-Math-17k-Processed",
    config: str = "en",
    test_size: float = 0.1,
    seed: int = 42,
):
    """
    Prepare and register DAPO-Math-17k datasets for training and testing.

    Args:
        dataset_name: HuggingFace dataset name
        config: Dataset configuration - "all" (17.4k), "cn" (3.28k), or "en" (14.1k)
        test_size: Fraction of data to use for testing
        seed: Random seed for train/test split

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    print(f"Loading dataset: {dataset_name} (config={config})")
    ds = load_dataset(dataset_name, config)["train"]
    print(f"Initial dataset length: {len(ds)}")

    # Map fields to standard format
    def map_fields(example):
        return {
            "question": example["prompt"],
            "final_answer": example["solution"],
            "data_source": example.get("data_source", "dapo_math"),
            "ability": example.get("ability", ""),
            "reward_model": example.get("reward_model", {}),
            "extra_info": example.get("extra_info", {}),
        }

    ds = ds.map(map_fields)
    print(f"After field mapping: {len(ds)}")

    # Train/test split
    ds_split = ds.train_test_split(test_size=test_size, seed=seed)
    train_ds, test_ds = ds_split["train"], ds_split["test"]
    print(f"Dataset split - train: {len(train_ds)}, test: {len(test_ds)}")

    # Register the datasets with the DatasetRegistry
    train_dataset = DatasetRegistry.register_dataset("dapo_math", train_ds, "train")
    test_dataset = DatasetRegistry.register_dataset("dapo_math", test_ds, "test")

    print(f"\nSuccessfully registered DAPO-Math datasets:")
    print(f"  Train: {len(train_dataset.get_data())} examples")
    print(f"  Test: {len(test_dataset.get_data())} examples")
    print(f"  Saved to: rllm/data/datasets/dapo_math/")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_dapo_math_data()

    # Display sample examples
    train_data = train_dataset.get_data()
    test_data = test_dataset.get_data()

    print("\nSample train example:")
    sample = train_data[0]
    print(f"  Question: {sample['question'][:100]}...")
    print(f"  Final answer: {sample['final_answer']}")
    print(f"  Data source: {sample.get('data_source', 'N/A')}")
    print(f"  Ability: {sample.get('ability', 'N/A')}")

    print("\nSample test example:")
    sample = test_data[0]
    print(f"  Question: {sample['question'][:100]}...")
    print(f"  Final answer: {sample['final_answer']}")
    print(f"  Data source: {sample.get('data_source', 'N/A')}")
    print(f"  Ability: {sample.get('ability', 'N/A')}")
