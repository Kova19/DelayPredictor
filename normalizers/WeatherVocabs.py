import json

import math


NUMERIC_FIELDS = [
    "temp",
    "visibility",
    "windSpeed",
    "humidity",
    "rain1H",
    "snow1H",
]

CATEGORICAL_FIELDS = [
    "group",
]


def load_history():
    with open("./normalizers/NewWeather.json", encoding="utf-8") as f:
        return json.load(f)


def build_numeric_scalers(data):
    scalers = {}

    for field in NUMERIC_FIELDS:
        values = [sample[field] for sample in data]

        mean = sum(values) / len(values)

        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)

        # ochrana proti dělení nulou
        if std == 0:
            std = 1.0

        scalers[field] = {
            "mean": mean,
            "std": std,
        }

    return scalers


def build_vocabs(data):
    vocabs = {}

    for field in CATEGORICAL_FIELDS:
        vocab = {"<UNK>": 0}

        unique = sorted({sample[field] for sample in data})

        for value in unique:
            vocab[value] = len(vocab)

        vocabs[field] = vocab

    return vocabs


def main():
    data = load_history()

    normalizer = {
        "numeric": build_numeric_scalers(data),
        "vocabs": build_vocabs(data),
    }

    with open("./normalizers/weatherNormalizer.json", "w", encoding="utf-8") as f:
        json.dump(normalizer, f, ensure_ascii=False, indent=2)

    print("Normalizer vytvořen.")

    print("\nNumeric:")
    for name, scaler in normalizer["numeric"].items():
        print(
            f"{name:12} mean={scaler['mean']:.3f} std={scaler['std']:.3f}"
        )

    print("\nVocabs:")
    for name, vocab in normalizer["vocabs"].items():
        print(f"{name:12} {len(vocab)-1} hodnot")


if __name__ == "__main__":
    main()
