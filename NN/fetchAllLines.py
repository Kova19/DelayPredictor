'''
Bachelor thesis FIT VUT
Author: Martin Kováčik (xkovacm01)
Date: 10.3.2026

Fetch script for fetching all lines, routes and stops from the API
'''

from datetime import datetime, timedelta
from typing import TypedDict

import requests
import json

from apiFolder.apiKeys import KEY, VALUE

urlRoutes = "https://dexter.fit.vutbr.cz/lissy/api/delayTrips/getAvailableRoutes"
urlTrips = "https://dexter.fit.vutbr.cz/lissy/api/delayTrips/getAvailableTrips"
urlForStops = "https://dexter.fit.vutbr.cz/lissy/api/shapes/getShape"

headers = {KEY: VALUE}


class Transports(TypedDict):
    line: str
    route: str
    vehicleType: int
    stop: str

# Get stops
def getStops(id: int):
    returnData: list = []

    try:
        params = {"shape_id": id}

        x = requests.get(urlForStops, headers=headers, params=params)
        obj = x.json()

        for n in obj["stops"]:
            returnData.append(n["stop_name"])

        return returnData

    except Exception as e:
        print(f"Error while getting stops: {e}")


# Get available trips
def getAvailableTrips(data, datesParam):
    transports: list[Transports] = []

    try:
        for n in data:
            params = {"dates": datesParam, "route_id": n["id"]}

            x = requests.get(urlTrips, headers=headers, params=params)
            obj = x.json()
            for j in obj:
                stops = getStops(j["shape_id"])
                for stop in stops:
                    fetchData = Transports(
                        line=n["route_short_name"],
                        route=j["stops"],
                        vehicleType=n["route_type"],
                        stop=stop
                    )
                    transports.append(fetchData)
        return transports

    except Exception as e:
        print(f"Error while getting trips: {e}")


# Get available routes for given dates
def getAvailableRoutes(datesParam):
    try:
        params = {
            "dates": datesParam,
        }
        x = requests.get(urlRoutes, headers=headers, params=params)
        obj = x.json()
        return getAvailableTrips(obj, datesParam)

    except Exception as e:
        print(f"Error while getting routes: {e}")


# Get all lines for training the normalizer
def getAllLinesForNorm():
    transports: list[Transports] = []

    for i in range(1, 8):
        date = datetime.today() - timedelta(days=i)
        print(date)

        year = date.year
        month = date.month - 1
        day = date.day

        datesParam = f'[["{year}-{month}-{day}","{year}-{month}-{day}"]]'

        newRecords = getAvailableRoutes(datesParam)

    transports = list({
        tuple(sorted(d.items())): d
        for d in transports + newRecords
    }.values())

    with open("AllLines2.json", "w", encoding="utf-8") as f:
        print(json.dumps(transports, indent=2, ensure_ascii=False), file=f)
    return transports


