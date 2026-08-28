import json
import os

from fetchers.fetchAllLines import getAllLinesForNorm


FIELDS = [
    "line",
    "route",
    "stop",
    "vehicleType",
]

VOCAB_PATH = "./normalizers/LinesVocab.json"


def build_vocab(data, fields):
    vocabs = {}

    for field in fields:
        vocab = {"<UNK>": 0}

        unique_values = sorted({str(item[field]) for item in data})

        for value in unique_values:
            vocab[value] = len(vocab)

        vocabs[field] = vocab

    return vocabs


def update_vocab(data, fields, existing_vocabs=None):
    if existing_vocabs is None:
        return build_vocab(data, fields)

    for field in fields:
        if field not in existing_vocabs:
            existing_vocabs[field] = {"<UNK>": 0}

        vocab = existing_vocabs[field]

        unique_values = sorted({str(item[field]) for item in data})

        for value in unique_values:
            if value not in vocab:
                vocab[value] = len(vocab)

    return existing_vocabs


def main():

    lines = getAllLinesForNorm()

    # Load existing vocab if available
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocabs = json.load(f)

        print("Existing vocab loaded")
        vocabs = update_vocab(lines, FIELDS, vocabs)

    else:
        print("No existing vocab found, building from scratch")
        vocabs = build_vocab(lines, FIELDS)

    # Save updated vocab
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocabs, f, ensure_ascii=False, indent=2)

    print("Vocab was updated\n")

    for name, vocab in vocabs.items():
        print(f"{name}: {len(vocab) - 1} values")


if __name__ == "__main__":
    main()
