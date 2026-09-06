import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from apiFolder.apiKeys import KEY, VALUE
from constants.constants import urlForRandomDelays
from fetchers.fetchAllWeather import fetchWeatherByDayNewEndpoint
from predict import predictDelay
import requests

headers = {KEY: VALUE}

availableMethods = ["neuralNetwork", "average", "randomForest", "linearRegression"]
MAX_WORKERS = 10


def predictTransport(transport, method, date, weather, weatherStations):
    predictObj = {
        "visualization": True,
        "date": date,
        "method": method,
        "depTime": transport["depTime"],
        "transport": {
            "line": transport["line"],
            "route": f"{transport['routeFrom']} -> {transport['routeTo']}"
        }
    }
    prediction = predictDelay(predictObj, weather, weatherStations)
    if prediction == (-1, -1, -1):
        return None

    return {
        "id": transport["id"],
        "line": transport["line"],
        "depTime": transport["depTime"],
        "routeFrom": transport["routeFrom"],
        "routeTo": transport["routeTo"],
        "delayData": prediction[2]
    }


def getPrediction(data, method, date):
    dateForWeather = datetime.strptime(date, "%Y-%m-%d")
    weather, weatherStations = fetchWeatherByDayNewEndpoint(dateForWeather)
    transports = data["trips"]
    totalTrips = len(data["trips"])
    results = [None] * len(transports)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                predictTransport,
                transport,
                method,
                date,
                weather,
                weatherStations,
            ): index
            for index, transport in enumerate(transports)
        }

        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            completed += 1
            print(f"Predicting {completed} of {totalTrips}")

    return [result for result in results if result is not None]


def getRandomTransports(date):
    print('Getting random delays for date: ', date)
    params = {
        'date': date
    }

    x = requests.get(urlForRandomDelays, headers=headers, params=params)
    fetchedDelays = x.json()

    for method in availableMethods:
        predictedDelays = getPrediction(fetchedDelays, method, date)
        with open(f"./simulations/{date}-{method}.json", "w", encoding="utf-8") as f:
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
