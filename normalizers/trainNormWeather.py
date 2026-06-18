'''
Bachelor thesis FIT VUT
Author: Martin Kováčik (xkovacm01)
Date: 5.3.2026

Train normalizer for weather data. This normalizer is used to encode the weather for NN
'''



import json
from typing import TypedDict

import joblib
#  Libraries for norm
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class weather(TypedDict):
    temp: float
    visibility: int
    windSpeed: float
    humidity: int
    rain1H: float
    snow1H: float
    group: str


def loadHistory() -> list[weather]:
    try:
        with open("weatherHistory.json") as data:
            loadJson = json.load(data)
            data.close()

        return loadJson

    except Exception as e:
        print(f"Error while loading json: {e}")
        exit(-1)

# main function for training the normalizer for weather data
def main():
    historyData = loadHistory()
    try:
        data = np.array(
            [
                [
                    weather["temp"],
                    weather["visibility"],
                    weather["windSpeed"],
                    weather["humidity"],
                    weather["rain1H"],
                    weather["snow1H"],
                    weather["group"],
                ]
                for weather in historyData
            ],
            dtype=object,
        )

        rowsForNumber = [0, 1, 2, 3, 4, 5]
        rowsForString = [6]

        preprocessor = ColumnTransformer(
            transformers=[
                ("numbers", StandardScaler(), rowsForNumber),
                ("category", OneHotEncoder(handle_unknown="ignore"), rowsForString),
            ]
        )
        preprocessor.fit(data)

        joblib.dump(preprocessor, "normalizers/weatherNormalizer.joblib")
        print("Weather normalizer succesfully trained")

    except Exception as e:
        print(f"Error while normalizing weather: {e}")
        exit(-1)


if __name__ == "__main__":
    main()
