import os
import json
import random
import numpy as np
import torch


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_GPQA_multiple_choice_answers(data):
    answers = [
        data["Correct Answer"],
        data["Incorrect Answer 1"],
        data["Incorrect Answer 2"],
        data["Incorrect Answer 3"],
    ]
    random.shuffle(answers)

    # Map options to letters
    options = ["A", "B", "C", "D"]
    options_to_answers = {letter: answer for letter, answer in zip(options, answers)}

    # Format the options into the string
    multiple_choice_string = ", ".join(
        f"{letter}) {options_to_answers[letter]}" for letter in options
    )

    # Save the letter corresponding to the correct answer
    correct_answer_letter = next(
        letter
        for letter, answer in options_to_answers.items()
        if answer == data["Correct Answer"]
    )

    return multiple_choice_string, correct_answer_letter


def _load_gpqa_diamond():
    import datasets

    loaded = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    train_data = loaded["train"].to_pandas()
    rows = [row.to_dict() for _, row in train_data.iterrows()]

    for problem in rows:
        multiple_choice_string, correct_answer_letter = (
            get_GPQA_multiple_choice_answers(problem)
        )
        problem["problem"] = (
            "Return your final response within \\boxed{{}} and only include the letter choice "
            "(A, B, C, or D) as your final response. "
            + problem["Question"]
            + "\n"
            + multiple_choice_string
        )
        problem["answer"] = correct_answer_letter

    return rows


def _load_jsonl_dataset(dataset_name):
    data_path = os.path.join(
        os.path.dirname(__file__), f"data/{dataset_name}/test.jsonl"
    )
    return load_jsonl(data_path)


def load_dataset(dataset_name):
    dataset_loaders = {
        "GPQADiamond": _load_gpqa_diamond,
        "amc23": lambda: _load_jsonl_dataset("amc23"),
        "aime24": lambda: _load_jsonl_dataset("aime24"),
        "math500": lambda: _load_jsonl_dataset("math500"),
    }

    if dataset_name not in dataset_loaders:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    return dataset_loaders[dataset_name]()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
