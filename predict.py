'''
Bachelor thesis FIT VUT
Author: Martin Kováčik (xkovacm01)
Date: 21.4.2026

Predict script for predicting delays based on the input data.
'''

import json
import sys
from datetime import datetime, timedelta

import requests
import torch

from apiFolder.apiKeys import KEY, KEYWEATHER, VALUE, BENWEATHER
from encoding import newEncode
from getShapeAndTripID import getShapeAndDelay
from fetchers.fetchAllWeather import decodeGroup
from fetchers.fetchDelays import firstPeak, isHoliday, secondPeak
from NN.neuralNetwork import DelayPredictor
from constants.constants import urlBenWeather, urlForWeather, urlForShape

headers = {KEY: VALUE}


with open("normalizers/weatherNormalizer.json", encoding="utf8") as f:
    weatherVocab = json.load(f)

with open("normalizers/LinesVocab.json", encoding="utf8") as f:
    linesVocab = json.load(f)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def getVocabSizes():
    with open("normalizers/LinesVocab.json", encoding="utf8") as f:
        linesVocab = json.load(f)

    return {name: len(mapping) for name, mapping in linesVocab.items()}


# Fetch weather data for given coordinates
def fetchWeather(coords):
    try:
        lat, lon = coords
        params = {"lat": lat, "lon": lon, "APPID": KEYWEATHER}
        x = requests.get(urlForWeather, params=params)
        obj = x.json()

        return obj

    except Exception as e:
        print(f"Error while fetching weather: {e}")
        return {}


# Find nearest meteorological station for given coordinates
def findNearMeteoStat(stopCords):
    now = datetime.now()
    nowMinus10 = now - timedelta(minutes=10)

    format = "%Y-%m-%dT%H:%M:%S"
    point = {"lat": stopCords[0], "lng": stopCords[1]}
    try:
        headers = {KEY: BENWEATHER}
        params = {
            "dateFrom": nowMinus10.strftime(format),
            "dateTo": now.strftime(format),
            "point": json.dumps(point),
        }
        x = requests.get(url=urlBenWeather, headers=headers, params=params)
        obj = x.json()

        if obj == []:
            print("No data found")
            raise ValueError("Data not found")

        return (obj[0]["coord"]["lat"], obj[0]["coord"]["lon"])

    except Exception as e:
        #  Print error and return Brno as fallback
        print(f"Error while finding near meteo stat: {e}")
        return [49.1952, 16.608]


# Parse weather data to the format needed for prediction
def parseWeather(weatherData):
    mainData = weatherData["main"]
    parsedObj = {
        "temp": mainData["temp"],
        "visibility": weatherData["visibility"],
        "windSpeed": weatherData["wind"]["speed"],
        "humidity": mainData["humidity"],
        "snow1H": weatherData.get("rain", {}).get("1h", 0.0),
        "rain1H": weatherData.get("rain", {}).get("1h", 0.0),
        "group": decodeGroup(weatherData["weather"][0]["id"]),
    }
    return parsedObj


# Find the best weather data for given date
def findBestTime(rawWeather, dateStr, timeStr):
    targetDt = datetime.strptime(f"{dateStr} {timeStr}", "%Y-%m-%d %H:%M:%S")

    targetTimestamp = int(targetDt.timestamp())

    nearest = min(
        rawWeather["list"], key=lambda item: abs(item["dt"] - targetTimestamp)
    )
    weather = parseWeather(nearest)
    return weather


# Get stop info for given shapeID
def getStopInfo(shapeID: int, stopLook: str):
    try:
        params = {"shape_id": shapeID}

        x = requests.get(urlForShape, params=params, headers=headers)
        obj = x.json()

        stops = obj["stops"]

        stopsArray = []

        for stop in stops:
            stopsArray.append(stop["stop_name"])

        i = 0
        for stop in stops:
            i += 1
            if stop["stop_name"] == stopLook:
                coords = stop["coords"]
                break

        if i == 0:
            return -1, -1
        else:
            returnObj = {"stopIndex": i, "stopCount": len(
                stops), "stops": stopsArray}
            return returnObj, coords

    except Exception as e:
        print(f"Error while fetching shape: {e}")
        return -1, -1


# Main function to predict delay for given input data
def predictDelay(data):

    realtimePrediction = -1
    usedRealtimeData = False

    # Parse arguments
    visualize = data["visualization"]
    predictingDate = data["date"]
    predictingDay = datetime.strptime(f"{predictingDate}", "%Y-%m-%d")
    depTime = data["depTime"]
    transport = data["transport"]
    shapeID, avgDelay, vehicleType, realtimeDelay = getShapeAndDelay(
        transport, depTime, predictingDate)

    if all(v == -1 for v in (shapeID, avgDelay, vehicleType, realtimeDelay)):
        return -1, -1, -1
    if all(v == -2 for v in (shapeID, avgDelay, vehicleType, realtimeDelay)):
        return -2, -2, -2

    if avgDelay == -10:
        return -3, -3, -3

    realtimePrediction = len(realtimeDelay) if realtimeDelay is not None else 0
    if realtimePrediction > 0:
        usedRealtimeData = True

    # Check if the object doesnt have stop and visualize is False
    if transport.get("stop") is None:
        if visualize is False:
            return -22

        predictionStop = transport["route"].split("->")[0].strip()
    else:
        predictionStop = transport["stop"]

    stopsInfo, stopCoords = getStopInfo(shapeID, predictionStop)

    destMeteo = findNearMeteoStat(stopCoords)
    weatherRaw = fetchWeather(destMeteo)
    weather = findBestTime(weatherRaw, predictingDate, depTime)

    if avgDelay == -10:
        return -10

    model = DelayPredictor(vocab_sizes=getVocabSizes()).to(device)
    state_dict = torch.load(
        "./models/delayPredictorModelV15.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    individualDelay = -1
    stopIndex = 1

    delay = {}
    prevDelay = 0
    realtimeBool = False
    for stop in stopsInfo["stops"][1:]:

        if realtimePrediction > 0:
            realtimeBool = True
            realtimePrediction -= 1
            result = realtimeDelay[stopIndex - 1]
            prevDelay = result

        else:
            realtimeBool = False
            transportInfo = {
                "line": transport["line"],
                "route": transport["route"],
                "vehicleType": vehicleType,
                "stop": stop,
            }
            tmp = {
                "dayOfWeek": f"{predictingDay.strftime('%A')}",
                "depTime": f"{depTime}",
                "7:00-8:30": firstPeak(depTime),
                "15:30-17:30": secondPeak(depTime),
                "transport": transportInfo,
                "stopIndex": stopIndex,
                "stopsCount": (stopsInfo["stopCount"] - 1),
                "weather": weather,
                "holiday": isHoliday(predictingDate),
                "prevDelay": prevDelay,
                "avgDelay": avgDelay[stopIndex - 1],
            }

            predictTensor = newEncode(
                    tmp,
                    True,
                    linesVocab,
                    weatherVocab
                )

            X = predictTensor.to(device).unsqueeze(0)
            result = model(X)
            prevDelay = round(result.item())

        returnResult = result if realtimeBool else result.item()

        if visualize:
            delay[str(stopIndex - 1)] = round(returnResult)
        elif predictionStop == stop:
            individualDelay = round(returnResult)
        stopIndex += 1

    if visualize:
        return shapeID, usedRealtimeData, delay
    else:
        return individualDelay


def main():
    inputArgs = json.loads(sys.argv[1])
    print(predictDelay(inputArgs))


if __name__ == "__main__":
    main()
