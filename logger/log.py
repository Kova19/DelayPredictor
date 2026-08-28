'''
Author: Martin Kováčik
'''

from datetime import datetime


def log(text, oneLine, model, version, mode="a"):
    print(f"{text}", end=", ") if oneLine else print(f"{text}")
    with open(
        f"./trainlogs/trainlog-{model}-{version}--{datetime.today().strftime("%Y-%m-%d")}.log",
        "a",
        encoding="utf-8",
    ) as f:
        print(f"{text}", end=", ", file=f) if oneLine else print(f"{text}", file=f)


