import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

from apiFolder.apiKeys import KEY, VALUE
from constants.constants import urlForRandomDelays
from fetchers.fetchAllWeather import fetchWeatherByDayNewEndpoint
from predict import predictDelayAllMethods

headers = {KEY: VALUE}

availableMethods = ["neuralNetwork", "average", "randomForest", "linearRegression"]
MAX_WORKERS = 10


def predictTransport(transport, date, weather, weatherStations):
    predictObj = {
        "visualization": True,
        "date": date,
        "depTime": transport["depTime"],
        "transport": {
            "line": transport["line"],
            "route": f"{transport['routeFrom']} -> {transport['routeTo']}",
        },
    }
    avgDelay, neuralNetwork, linearRegression, randomForest = predictDelayAllMethods(
        predictObj, weather, weatherStations
    )
    if (
        avgDelay == -1
        or neuralNetwork == -1
        or linearRegression == -1
        or randomForest == -1
    ):
        return None
    if (
        avgDelay == -2
        or neuralNetwork == -2
        or linearRegression == -2
        or randomForest == -2
    ):
        return None
    if (
        avgDelay == -3
        or neuralNetwork == -3
        or linearRegression == -3
        or randomForest == -3
    ):
        return None

    retAvgDelay = {
        "id": transport["id"],
        "line": transport["line"],
        "depTime": transport["depTime"],
        "routeFrom": transport["routeFrom"],
        "routeTo": transport["routeTo"],
        "delayData": avgDelay,
    }

    retNeuralNetwork = {
        "id": transport["id"],
        "line": transport["line"],
        "depTime": transport["depTime"],
        "routeFrom": transport["routeFrom"],
        "routeTo": transport["routeTo"],
        "delayData": neuralNetwork,
    }

    retLinearRegression = {
        "id": transport["id"],
        "line": transport["line"],
        "depTime": transport["depTime"],
        "routeFrom": transport["routeFrom"],
        "routeTo": transport["routeTo"],
        "delayData": linearRegression,
    }

    retRandomForest = {
        "id": transport["id"],
        "line": transport["line"],
        "depTime": transport["depTime"],
        "routeFrom": transport["routeFrom"],
        "routeTo": transport["routeTo"],
        "delayData": randomForest,
    }

    return retAvgDelay, retNeuralNetwork, retLinearRegression, retRandomForest


def getPrediction(data, date):
    dateForWeather = datetime.strptime(date, "%Y-%m-%d")
    weather, weatherStations = fetchWeatherByDayNewEndpoint(dateForWeather)
    transports = data
    totalTrips = len(data)
    results = [None] * len(transports)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                predictTransport,
                transport,
                date,
                weather,
                weatherStations,
            ): index
            for index, transport in enumerate(transports)
        }

        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as error:
                transport = transports[index]
                print(f"Prediction failed for transport {
                        transport.get('id')}: {error}")
            completed += 1
            print(f"Predicting {completed} of {totalTrips}")

    predictions = [result for result in results if result is not None]
    return tuple(
        [prediction[index] for prediction in predictions] for index in range(4)
    )


def getRandomTransports(date):
    delays = []

    params = {"date": date}

    x = requests.get(urlForRandomDelays, headers=headers, params=params)
    fetchedDelays = x.json()

    delays.extend(fetchedDelays["trips"])
    while len(delays) < (fetchedDelays["stats"]["number_of_trips"] * 0.6):
        # while(len(delays) < 5000):
        params = {"date": date, "tripIdFrom": fetchedDelays["trips"][-1]["id"]}
        x = requests.get(urlForRandomDelays, headers=headers, params=params)
        fetchedDelays = x.json()
        delays.extend(fetchedDelays["trips"])

    # delays = [delays[4777], delays[4778], delays[4779], delays[4780], delays[4781], delays[4782], delays[4783], delays[4784], delays[4785], delays[4786]]
    avgDelay, neuralNetwork, linearRegression, randomForest = getPrediction(
        delays, date
    )
    with open(f"./simulations/{date}-avgDelay.json", "w", encoding="utf-8") as f:
        print(json.dumps(avgDelay, indent=2, ensure_ascii=False), file=f)
    with open(f"./simulations/{date}-neuralNetwork.json", "w", encoding="utf-8") as f:
        print(json.dumps(neuralNetwork, indent=2, ensure_ascii=False), file=f)
    with open(
        f"./simulations/{date}-linearRegression.json", "w", encoding="utf-8"
    ) as f:
        print(json.dumps(linearRegression, indent=2, ensure_ascii=False), file=f)
    with open(f"./simulations/{date}-randomForest.json", "w", encoding="utf-8") as f:
        print(json.dumps(randomForest, indent=2, ensure_ascii=False), file=f)


# Argument parser
def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--days", help="Write day for predict", type=str)

    return parser.parse_args()


def main():
    args = parseArguments()

    getRandomTransports(args.days)


if __name__ == "__main__":
    main()
