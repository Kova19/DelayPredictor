'''
Bachelor thesis FIT VUT
Author: Martin Kováčik (xkovacm01)
Date: 5.3.2026

Train normalizer for lines, routes and stops. This normalizer is used to encode
the categorical features of lines, routes and stops into numerical values
that can be used by the neural network.
'''

import json
from typing import TypedDict

import joblib
#  Libraries for norm
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from NN.fetchAllLines import getAllLinesForNorm


class Line(TypedDict):
    line: str
    route: str
    vehicleType: int
    stop: str


# This is legacy code only for debugging
def loadLines() -> list[Line]:
    try:
        with open("normalizers/AllLines2.json") as data:
            loadJson = json.load(data)
            data.close()
        return loadJson

    except Exception as e:
        print(f"Error while loading json: {e}")
        exit(-1)


# Main function for training the normalizer for lines, routes, vehicleType and stops
def main():
    lines = getAllLinesForNorm()

    try:
        data = np.array(
            [
                [
                    line["line"],
                    line["route"],
                    line["vehicleType"],
                    line["stop"],
                ]
                for line in lines
            ],
            dtype=object,
        )

        rowsLines = [0, 1, 2, 3]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "linesCategory",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    rowsLines,
                )
            ]
        )

        preprocessor.fit(data)
        joblib.dump(preprocessor, "normalizers/linesNormalizer2.joblib")
        print("Lines normalizer succesufully trained")

    except Exception as e:
        print(f"Error while normalizing lines: {e}")
        exit(-1)


if __name__ == "__main__":
    main()
