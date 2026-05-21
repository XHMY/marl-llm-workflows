"""
Prepare and register DeepMath-103K dataset for multi-agent math reasoning training.

This script:
1. Downloads DeepMath-103K from HuggingFace
2. Filters by difficulty range (4-7)
3. Filters to keep only integer-convertible answers
4. Splits into train/test sets
5. Saves to parquet files in rllm/data/datasets/deepmath/
6. Registers with DatasetRegistry
"""

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


def is_convertible_to_int(final_answer: str) -> bool:
    """Check if final answer can be converted to integer."""
    final_str = str(final_answer).strip()
    if not final_str:
        return False
    if final_str.isdigit() or (final_str.startswith('-') and final_str[1:].isdigit()):
        return True
    return False


def prepare_deepmath_data(
    dataset_name: str = "zwhe99/DeepMath-103K",
    difficulty_range: tuple[int, int] = (4, 7),
    filter_convertible_to_int: bool = True,
    test_size: float = 0.01,
    seed: int = 42,
):
    """
    Prepare and register DeepMath-103K datasets for training and testing.

    Args:
        dataset_name: HuggingFace dataset name
        difficulty_range: (min_difficulty, max_difficulty) tuple
        filter_convertible_to_int: Whether to filter for integer-convertible answers
        test_size: Fraction of data to use for testing
        seed: Random seed for train/test split

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)["train"]
    print(f"Initial dataset length: {len(ds)}")

    # Filter by difficulty
    if difficulty_range:
        min_diff, max_diff = difficulty_range
        ds = ds.filter(lambda x: min_diff <= x["difficulty"] <= max_diff)
        print(f"After difficulty filter [{min_diff}-{max_diff}]: {len(ds)}")

    # Filter to keep only entries where final_answer can be converted to int
    if filter_convertible_to_int:
        ds = ds.filter(lambda x: is_convertible_to_int(x["final_answer"]))
        print(f"After integer answer filter: {len(ds)}")

    # Train/test split
    ds_split = ds.train_test_split(test_size=test_size, seed=seed)
    train_ds, test_ds = ds_split["train"], ds_split["test"]
    print(f"Dataset split - train: {len(train_ds)}, test: {len(test_ds)}")

    # Register the datasets with the DatasetRegistry
    train_dataset = DatasetRegistry.register_dataset("deepmath", train_ds, "train")
    test_dataset = DatasetRegistry.register_dataset("deepmath", test_ds, "test")

    print(f"\nSuccessfully registered DeepMath datasets:")
    print(f"  Train: {len(train_dataset.get_data())} examples")
    print(f"  Test: {len(test_dataset.get_data())} examples")
    print(f"  Saved to: rllm/data/datasets/deepmath/")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_deepmath_data(difficulty_range=[8,9])

    # Display sample examples
    train_data = train_dataset.get_data()
    test_data = test_dataset.get_data()

    print("\nSample train example:")
    sample = train_data[0]
    print(f"  Question: {sample['question'][:100]}...")
    print(f"  Final answer: {sample['final_answer']}")
    print(f"  Difficulty: {sample.get('difficulty', 'N/A')}")

    print("\nSample test example:")
    sample = test_data[0]
    print(f"  Question: {sample['question'][:100]}...")
    print(f"  Final answer: {sample['final_answer']}")
    print(f"  Difficulty: {sample.get('difficulty', 'N/A')}")
