import random

from datasets import load_dataset
from rllm.data.dataset import DatasetRegistry


def prepare_gpqa_data():
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    def preprocess_fn(example, idx):
        correct = example["Correct Answer"]
        distractors = [
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]
        choices = [correct] + distractors

        rng = random.Random(idx)
        rng.shuffle(choices)

        correct_letter = chr(ord("A") + choices.index(correct))

        letters = ["A", "B", "C", "D"]
        choices_text = "\n".join(
            f"{letter}) {choice}" for letter, choice in zip(letters, choices)
        )
        question = f"{example['Question']}\n\n{choices_text}"

        return {
            "question": question,
            "final_answer": correct_letter,
            "data_source": "gpqa",
        }

    dataset = dataset.map(preprocess_fn, with_indices=True)
    dataset = DatasetRegistry.register_dataset("gpqa_diamond", dataset, "test")
    return dataset


if __name__ == "__main__":
    dataset = prepare_gpqa_data()
    print(dataset.get_data_path())
