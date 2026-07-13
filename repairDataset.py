import json
from pathlib import Path
from encoding import newEncode
import torch
import argparse


DATASET_DIR = Path(__file__).resolve().parent / "dataset"

days = [p.name for p in DATASET_DIR.iterdir() if p.is_dir()]
with open("normalizers/weatherNormalizer.json", encoding="utf8") as f:
    weatherVocab = json.load(f)

with open("normalizers/LinesVocab.json", encoding="utf8") as f:
    linesVocab = json.load(f)


def renormalizeDataset(dataset_dir: Path) -> dict:

    for day in days:
        path = dataset_dir / day
        print(f"Renormalizing dataset for day: {day}")

        print(f"json count: {len(list(path.glob('*.json')))}")
        dataset = []
        for json_file in path.glob("*.json"):
            print(f"Renormalizing file: {json_file}")
            with json_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)

            tensors = [newEncode(data, False, linesVocab, weatherVocab) for data in data]
            tensors = [t for t in tensors if t is not None]

            if tensors:
                dataset.append(torch.stack(tensors))

        if not dataset:
            print(f"Error: No valid encoded records for day: {day}, skipping dataset.pt update")
            continue

        day_tensor = torch.cat(dataset)
        torch.save(day_tensor, dataset_dir / day / "dataset.pt")


def fixAndLoadDataset(dataset_dir: Path) -> dict:

    for day in days:
        path = dataset_dir / day
        print(f"Repairing dataset for day: {day}")

        print(f"json count before repair: {len(list(path.glob('*.json')))}")
        dataset = []
        for json_file in path.glob("*.json"):
            print(f"Repairing file: {json_file}")
            with json_file.open("r", encoding="utf-8") as handle:
                dataForFix = json.load(handle)

            fixedCopy = []
            for data in dataForFix:
                if data["delay"] < -5 or data["delay"] > 90:
                    continue
                if data["weather"]["visibility"] is None:
                    fixedWeather = {
                        "temp": data["weather"]["temp"],
                        "visibility": 10000,
                        "windSpeed": data["weather"]["windSpeed"],
                        "humidity": data["weather"]["humidity"],
                        "snow1H": data["weather"]["snow1H"],
                        "rain1H": data["weather"]["rain1H"],
                        "group": data["weather"]["group"],
                    }
                record = {
                    'dayOfWeek': data['dayOfWeek'],
                    'depTime': data['depTime'],
                    '7:00-8:30': data['7:00-8:30'],
                    '15:30-17:30': data['15:30-17:30'],
                    'transport': data['transport'],
                    'stopIndex': data['stopIndex'],
                    'stopsCount': data['stopsCount'],
                    'weather': fixedWeather if data["weather"]["visibility"] is None else data['weather'],
                    'holiday': data['holiday'],
                    'prevDelay': data['prevDelay'],
                    'avgDelay': data['avgDelay'],
                    'delay': data['delay']
                }
                fixedCopy.append(record)

            tensors = [newEncode(data, False, linesVocab, weatherVocab) for data in data]
            tensors = [t for t in tensors if t is not None]

            if tensors:
                dataset.append(torch.stack(tensors))

        if not dataset:
            print(f"No valid encoded records for day: {day}, skipping dataset.pt update")
            continue

        day_tensor = torch.cat(dataset)
        torch.save(day_tensor, dataset_dir / day / "dataset.pt")


# Argument parser
def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--mode", help="How this script should work 'fix' or 'normalize'", type=str
    )

    return parser.parse_args()


def main():
    if not DATASET_DIR.exists():
        print(f"Dataset folder does not exists: {DATASET_DIR}")
        return

    args = parseArguments()

    if args.mode == "fix":
        fixAndLoadDataset(DATASET_DIR)
    elif args.mode == "normalize":
        renormalizeDataset(DATASET_DIR)
    else:
        print("Wrong mode")
        return


if __name__ == "__main__":
    main()
