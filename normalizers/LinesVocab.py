import json
from fetchers.fetchAllLines import getAllLinesForNorm


# Main object for transport info
FIELDS = [
    "line",
    "route",
    "stop",
    "vehicleType",
]


def build_vocab(data, fields):
    vocabs = {}

    for field in fields:
        vocab = {"<UNK>": 0}

        # Set for unique values
        unique_values = sorted({str(item[field]) for item in data})

        for value in unique_values:
            vocab[value] = len(vocab)

        vocabs[field] = vocab

    return vocabs


def main():

    lines = getAllLinesForNorm()

    vocabs = build_vocab(lines, FIELDS)

    with open("./normalizers/LinesVocab.json", "w", encoding="utf-8") as f:
        json.dump(vocabs, f, ensure_ascii=False, indent=2)

    print("Vocab was built\n")

    for name, vocab in vocabs.items():
        print(f"{name}: {len(vocab)-1} values")


if __name__ == "__main__":
    main()
