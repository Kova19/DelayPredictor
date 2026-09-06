import argparse
import json
from datetime import datetime
from apiFolder.apiKeys import KEY, VALUE
from constants.constants import urlForRandomDelays
from fetchers.fetchAllWeather import fetchWeatherByDayNewEndpoint
from predict import predictDelay
import requests

headers = {KEY: VALUE}

availableMethods = ["neuralNetwork", "average", "randomForest", "linearRegression"]


def getPrediction(data, method, date):
    dateForWeather = datetime.strptime(date, "%Y-%m-%d")
    weather, weatherStations = fetchWeatherByDayNewEndpoint(dateForWeather)

    returnObj = []
    i = 0
    for transport in data["trips"]:
        i += 1
        print(f"Predicting {i} of {data["stats"]["number_of_trips"]}")
        predictObj = {
            "visualization": True,
            "date": date,
            "method": method,
            "depTime": transport["depTime"],
            "transport": {
                "line": transport["line"],
                "route": f"{transport["routeFrom"]} -> {transport["routeTo"]}"
            }
        }
        prediction = predictDelay(predictObj, weather, weatherStations)
        if prediction != (-1, -1, -1):
            tmp = {
                "id": transport["id"],
                "line": transport["line"],
                "depTime": transport["depTime"],
                "routeFrom": transport["routeFrom"],
                "routeTo": transport["routeTo"],
                "delayData": prediction[2]
            }
            returnObj.append(tmp)
    return returnObj


def getRandomTransports(date):
    print('Getting random delays for date: ', date)
    params = {
        'date': date
    }

    x = requests.get(urlForRandomDelays, headers=headers, params=params)
    fetchedDelays = x.json()

    for method in availableMethods:
        predictedDelays = getPrediction(fetchedDelays, method, date)
        with open(f"{date}-{method}.json", "w", encoding="utf-8") as f:
            print(json.dumps(predictedDelays, indent=2, ensure_ascii=False), file=f)


# Argument parser
def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--days", help="Write day for predict", type=str
    )

    return parser.parse_args()


def main():
    args = parseArguments()

    getRandomTransports(args.days)


if __name__ == "__main__":
    main()
