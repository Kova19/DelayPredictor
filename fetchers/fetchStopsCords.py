'''
Bachelor thesis FIT VUT
Author: Martin Kováčik (xkovacm01)
Date: 10.3.2026

Fetch script for fetching stop coordinates from the API.
'''

from typing import Tuple, TypedDict

import requests

from apiFolder.apiKeys import KEY, VALUE
from constants.constants import urlForShapes, urlForShape


class Stops(TypedDict):
    name: str
    cords: Tuple[float, float]


# Main functio for fetching stops
def getAllStops(dateStr: str):
    returnStops = []
    shapeIds = []
    names = []
    headers = {KEY: VALUE}
    params = {"date": dateStr}

    x = requests.get(urlForShapes, headers=headers, params=params)
    shapes = x.json()

    for shape in shapes:
        trips = shape["trips"]
        for trip in trips:
            shapeIds.append(trip["shape_id"])
    for id in shapeIds:
        params = {"shape_id": id}
        x = requests.get(urlForShape, headers=headers, params=params)
        shape = x.json()

        stops = shape["stops"]
        for stop in stops:
            tmp = {"StopName": stop["stop_name"], "coords": stop["coords"]}
            if tmp["StopName"] not in names:
                returnStops.append(tmp)
                names.append(tmp["StopName"])
    return returnStops
