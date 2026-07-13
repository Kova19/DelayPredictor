'''
Bachelor thesis FIT VUT
Author: Martin Kováčik (xkovacm01)
Date: 6.4.2026

Encoding script for encoding the data for the neural network.
'''

import math
import json
from typing import Tuple

import numpy as np
import torch

days = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


def toFloatArray(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def encodeTransport(transport, vocab):
    return [
        vocab["line"].get(str(transport["line"]), 0),
        vocab["route"].get(str(transport["route"]), 0),
        vocab["vehicleType"].get(str(transport["vehicleType"]), 0),
        vocab["stop"].get(str(transport["stop"]), 0),
    ]


def normalizeWeather(weather, vocab):
    out = []

    # Numeric values
    for field, scaler in vocab["numeric"].items():
        out.append((
            weather[field] - scaler["mean"]
        ) / scaler["std"])

    # Categorical values
    for field, vocab in vocab["vocabs"].items():
        out.append(vocab.get(weather[field], 0))

    return out


# Encode departure time to sine and cosine values to capture the cyclical nature of time
def encodeDepartTime(timeStr: str) -> Tuple[float, float]:
    hour, minute = map(int, timeStr.split(":")[:2])
    minutes = (hour * 60) + minute
    totalDayMinutes = 1440

    return (
        math.sin(2 * math.pi * minutes / totalDayMinutes),
        math.cos(2 * math.pi * minutes / totalDayMinutes),
    )


# Encode day of week same as departure time
def encodeDays(day: int) -> Tuple[float, float]:
    daysOfWeek = 7

    return (
        math.sin(2 * math.pi * day / daysOfWeek),
        math.cos(2 * math.pi * day / daysOfWeek),
    )


# Encode the input data
def newEncode(obj, inference, linesVocab, weatherVocab):
    try:
        # Parse input data from object
        parsedTransport = {
            "line": obj["transport"]["line"],
            "route": obj["transport"]["route"],
            "vehicleType": obj["transport"]["vehicleType"],
            "stop": obj["transport"]["stop"],
         }

        parsedWeather = {
            "temp": obj["weather"]["temp"],
            "visibility": obj["weather"]["visibility"],
            "windSpeed": obj["weather"]["windSpeed"],
            "humidity": obj["weather"]["humidity"],
            "rain1H": obj["weather"]["rain1H"],
            "snow1H": obj["weather"]["snow1H"],
            "group": obj["weather"]["group"]
        }

        # Normalize the transport data
        normTransport = encodeTransport(parsedTransport, linesVocab)
        normWeather = normalizeWeather(parsedWeather, weatherVocab)

        # Convert the parsed data to PyTorch tensors
        transportDetail = torch.tensor(normTransport, dtype=torch.float32)
        weather = torch.tensor(normWeather, dtype=torch.float32)

        departureTime = torch.tensor(
            encodeDepartTime(obj["depTime"]), dtype=torch.float32
        )
        dayOfWeek = torch.tensor(
            encodeDays(days[obj["dayOfWeek"]]), dtype=torch.float32
        )
        holiday = torch.tensor([1.0 if obj["holiday"] else 0.0])
        peakOne = torch.tensor([1.0 if obj["7:00-8:30"] else 0.0])
        peakTwo = torch.tensor([1.0 if obj["15:30-17:30"] else 0.0])

        stopIndex = torch.tensor([obj["stopIndex"]], dtype=torch.float32)
        stopCount = torch.tensor([obj["stopsCount"]], dtype=torch.float32)

        prevDelay = torch.tensor([obj["prevDelay"]], dtype=torch.float32)
        avgDelay = torch.tensor([obj["avgDelay"]], dtype=torch.float32)

        if (inference):
            return torch.cat(
                [
                    transportDetail,
                    weather,
                    dayOfWeek,
                    departureTime,
                    peakOne,
                    peakTwo,
                    stopIndex,
                    stopCount,
                    holiday,
                    prevDelay,
                    avgDelay,
                ]
            )
        else:
            delay = torch.tensor([obj["delay"]], dtype=torch.float32)
            return torch.cat(
                [
                    transportDetail,
                    weather,
                    dayOfWeek,
                    departureTime,
                    peakOne,
                    peakTwo,
                    stopIndex,
                    stopCount,
                    holiday,
                    prevDelay,
                    avgDelay,
                    delay,
                ]
            )

    except Exception as e:
        print(f"Erro while encoding obj: {e}")
        return None


# Encode the input data
def encode(obj, linesNormalizer, weatherNormalizer):
    try:
        tmpTransport = np.array(
            [
                [
                    obj["transport"]["line"],
                    obj["transport"]["route"],
                    obj["transport"]["vehicleType"],
                    obj["transport"]["stop"],
                ]
            ],
            dtype=object,
        )

        tmpWeather = np.array(
            [
                [
                    obj["weather"]["temp"],
                    obj["weather"]["visibility"],
                    obj["weather"]["windSpeed"],
                    obj["weather"]["humidity"],
                    obj["weather"]["rain1H"],
                    obj["weather"]["snow1H"],
                    obj["weather"]["group"],
                ]
            ],
            dtype=object,
        )

        normTransport = linesNormalizer.transform(tmpTransport)
        normWeather = weatherNormalizer.transform(tmpWeather)


        transportDetail = torch.from_numpy(normTransport).float().squeeze()
        departureTime = torch.tensor(
            encodeDepartTime(obj["depTime"]), dtype=torch.float32
        )
        weather = torch.from_numpy(normWeather).float().squeeze()
        dayOfWeek = torch.tensor(
            encodeDays(days[obj["dayOfWeek"]]), dtype=torch.float32
        )
        holiday = torch.tensor([1.0 if obj["holiday"] else 0.0])
        peakOne = torch.tensor([1.0 if obj["7:00-8:30"] else 0.0])
        peakTwo = torch.tensor([1.0 if obj["15:30-17:30"] else 0.0])

        stopIndex = torch.tensor([obj["stopIndex"]], dtype=torch.float32)
        stopCount = torch.tensor([obj["stopsCount"]], dtype=torch.float32)

        prevDelay = torch.tensor([obj["prevDelay"]], dtype=torch.float32)
        avgDelay = torch.tensor([obj["avgDelay"]], dtype=torch.float32)

        delay = torch.tensor([obj["delay"]], dtype=torch.float32)
        return torch.cat(
            [
                dayOfWeek,
                departureTime,
                peakOne,
                peakTwo,
                transportDetail,
                stopIndex,
                stopCount,
                weather,
                holiday,
                prevDelay,
                avgDelay,
                delay,
            ]
        )

    except Exception as e:
        print(f"Erro while encoding obj: {e}")
        print(obj)
        return None


def encodeForPrediction(obj, linesNormalizer, weatherNormalizer):
    try:
        tmpTransport = np.array(
            [
                [
                    obj["transport"]["line"],
                    obj["transport"]["route"],
                    obj["transport"]["vehicleType"],
                    obj["transport"]["stop"],
                ]
            ],
            dtype=object,
        )
        tmpWeather = np.array(
            [
                [
                    obj["weather"]["temp"],
                    obj["weather"]["visibility"],
                    obj["weather"]["windSpeed"],
                    obj["weather"]["humidity"],
                    obj["weather"]["rain1H"],
                    obj["weather"]["snow1H"],
                    obj["weather"]["group"],
                ]
            ],
            dtype=object,
        )

        normTransport = linesNormalizer.transform(tmpTransport)
        normWeather = weatherNormalizer.transform(tmpWeather)

        transportDetail = torch.from_numpy(normTransport).float().squeeze()
        departureTime = torch.tensor(
            encodeDepartTime(obj["depTime"]), dtype=torch.float32
        )
        weather = torch.from_numpy(normWeather).float().squeeze()
        dayOfWeek = torch.tensor(
            encodeDays(days[obj["dayOfWeek"]]), dtype=torch.float32
        )
        holiday = torch.tensor([1.0 if obj["holiday"] else 0.0])
        peakOne = torch.tensor([1.0 if obj["7:00-8:30"] else 0.0])
        peakTwo = torch.tensor([1.0 if obj["15:30-17:30"] else 0.0])

        stopIndex = torch.tensor([obj["stopIndex"]], dtype=torch.float32)
        stopCount = torch.tensor([obj["stopsCount"]], dtype=torch.float32)

        prevDelay = torch.tensor([obj["prevDelay"]], dtype=torch.float32)
        avgDelay = torch.tensor([obj["avgDelay"]], dtype=torch.float32)

        return torch.cat(
            [
                dayOfWeek,
                departureTime,
                peakOne,
                peakTwo,
                transportDetail,
                stopIndex,
                stopCount,
                weather,
                holiday,
                prevDelay,
                avgDelay,
            ]
        )

    except Exception as e:
        print(f"Erro while encoding obj: {e}")
        print(obj)
        return None


def main():
    obj = {
        "dayOfWeek": "Friday",
        "depTime": "06:31:00",
        "7:00-8:30": False,
        "15:30-17:30": False,
        "transport": {
          "line": "10",
          "route": "Komín, smyčka -> Novolíšeňská",
          "vehicleType": 0,
          "stop": "Vozovna Komín"
        },
        "stopIndex": 1,
        "stopsCount": 22,
        "weather": {
          "temp": 279.6,
          "visibility": 10000,
          "windSpeed": 0.99,
          "humidity": 34,
          "snow1H": 0.0,
          "rain1H": 0.0,
          "group": "Clouds"
        },
        "holiday": False,
        "prevDelay": 0,
        "avgDelay": 0,
        "delay": 0
    }

    print(newEncode(obj, False))

if __name__=="__main__":
    main()
