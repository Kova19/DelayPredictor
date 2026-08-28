'''
Author: Martin Kováčik

Script for fetch shape ID and avgDelays for given transport and departure time.
'''

from datetime import datetime, timedelta

import requests
import numpy as np
from zoneinfo import ZoneInfo

from apiFolder.apiKeys import KEY, VALUE, BENWEATHER
from constants.constants import urlForRealtimeDelays, urlForShape, urlForAvgDelays
from fetchers.fetchDelays import fixDelays
from geopy.distance import geodesic

headers = {KEY: VALUE}
headersBen = {KEY: BENWEATHER}


# Count stops for given shapeID
def countStops(shapeID):
    try:
        params = {
            "shape_id": shapeID
        }

        x = requests.get(urlForShape, headers=headers, params=params)
        obj = x.json()

        return len(obj["stops"])

    except Exception as e:
        print(e)
        return -1


# Convert departure time to UTC format for given time in local timezone
def convertUTCTime(depTime):
    depTime = datetime.strptime(depTime, '%H:%M:%S')
    today = datetime.today().date()

    depTime = depTime.replace(year=today.year, month=today.month, day=today.day)

    depTime = depTime.replace(tzinfo=ZoneInfo("Europe/Prague"))

    returnDateTime = depTime.astimezone(ZoneInfo("UTC"))

    return returnDateTime.strftime("%Y-%m-%dT%H:%M:%S.000Z")


# Try to find realtime delays if its possible
def getRealtimeDelays(benRouteID, key, dateFrom):
    dateTo = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    params = {
        'key': key,
        'uidFrom': 0,
        'fields': '["ben", "RouteID", "Latitude", "Longitude", "DelayInMins"]',
        'dateFrom': dateFrom,
        'dateTo': dateTo
    }

    filteredData = []
    try:
        x = requests.get(urlForRealtimeDelays, params=params, headers=headersBen)
        obj = x.json()

        for data in obj:
            if data["ben"]["key"] == key and data["RouteID"] == benRouteID:
                delay = {
                    'coords': (data["Latitude"], data["Longitude"]),
                    'delay': int(data["DelayInMins"]),
                }
                filteredData.append(delay)

        uniqueData = list({tuple(d['coords']): d for d in filteredData}.values())
        return uniqueData

    except Exception as e:
        print(e)
        return None


# Check if realtime prediction is possible 
def isRealTimePrediction(predictionDay, depTime, benID, lineID):
    if not benID:
        return False, None

    depTimeUTC = convertUTCTime(depTime)

    depTimePrediction = datetime.strptime(depTimeUTC, "%Y-%m-%dT%H:%M:%S.000Z")

    predDate = datetime.strptime(predictionDay, "%Y-%m-%d")
    predDate = predDate.replace(hour=depTimePrediction.hour, minute=depTimePrediction.minute, second=depTimePrediction.second)

    today = datetime.utcnow()
    thresHoldTime = predDate + timedelta(hours=2)

    predDate = datetime.fromisoformat(depTimeUTC.replace("Z", "+00:00")).replace(tzinfo=None)

    if predDate < today < thresHoldTime:
        actualVehiclePositions = getRealtimeDelays(benID, lineID, predDate)

        return True, actualVehiclePositions
    else:
        return False, None


# Map realtime delays to stops
def mapDelays(rawDelays, shapeID):
    params = {
        "shape_id": shapeID
    }

    x = requests.get(urlForShape, headers=headers, params=params)
    shapeDetail = x.json()
    delays = []

    distance = geodesic(shapeDetail["stops"][-1]['coords'], rawDelays[-1]['coords']).meters
    if distance < 10:
        return None, False

    for stop in shapeDetail["stops"][1:]:
        bestMetrs = float("inf")

        for raw in rawDelays:
            tmp = geodesic(stop['coords'], raw['coords']).meters
            if tmp < bestMetrs:
                nearestRaw = raw
                bestMetrs = tmp
        if bestMetrs > 100:
            break
        else:
            delays.append(nearestRaw["delay"])
    return delays, True


# Get average delays for given tripID
def getMedianDelays(data, sectionCnt):
    try:
        #  return {} if data are not provided
        if not data:
            return {}

        result = {}
        i = 0

        while i < sectionCnt:  # bcs index from 0
            values = []

            # Get all data from current section
            for days in data:
                for day in days:
                    delays = days[day]

                    sectionData = delays.get(str(i))
                    if sectionData is not None:
                        values.append(sectionData)

            # Make median and use np.nan for empty values
            if len(values) == 0:
                if i == 0:
                    # Except that vehicle will start with 0 delay
                    result[i] = 0
                else:
                    # If values is missing use np.nan
                    result[i] = np.nan
            else:
                #  Calculate median
                tmp = np.median(values)
                #  Check if its np.nan if its save np.nan else save median
                result[i] = np.nan if np.isnan(tmp) else int(tmp)
            i += 1
        fixedResults = fixDelays(result)

        return fixedResults

    except Exception as e:
        print(f"Error while getting avgdelays: {e}")
        raise


# Get average delays for given tripID
def getAvgDelaysForPred(data, sectionCnt):
    try:
        #  return {} if data are not provided
        if not data:
            return {}

        result = {}
        i = 0

        while i < sectionCnt:  # bcs index from 0
            values = []
            # Get all data from current section
            for days in data:
                for day in days:
                    delays = days[day]

                    sectionData = delays.get(str(i))
                    if sectionData is not None:
                        values.append(sectionData)

            # Make median and use np.nan for empty values
            if len(values) == 0:
                if i == 0:
                    # Except that vehicle will start with 0 delay
                    result[i] = 0
                else:
                    # If values is missing use np.nan
                    result[i] = np.nan
            else:
                #  Calculate avg
                tmp = np.average(values)
                #  Check if its np.nan if its save np.nan else save median
                result[i] = np.nan if np.isnan(tmp) else int(tmp)
            i += 1
        fixedResults = fixDelays(result)

        return fixedResults

    except Exception as e:
        print(f"Error while getting avgdelays: {e}")
        raise


def getDataAboutTransport(transport, depTime, predictionDay, avgDelay):
    stationFrom, stationTo = transport["route"].split(" -> ")
    jsPredictionDay = datetime.strptime(predictionDay, "%Y-%m-%d")
    jsPredictionDay = f"{jsPredictionDay.year}-{jsPredictionDay.month - 1}-{jsPredictionDay.day}"

    params = {
        "line": transport["line"],
        "routeTo": stationTo,
        "routeFrom": stationFrom,
        "depTime": depTime,
        "date": jsPredictionDay,
        "weeks": 12 if avgDelay else 8
    }

    x = requests.get(urlForAvgDelays, headers=headers, params=params)
    fetchedDelays = x.json()

    if fetchedDelays == []:
        return -1, -1, -1, -1, -1

    # Get shape ID for stop count and vehicleType
    shapeID = fetchedDelays["shape_id"]

    stopCount = countStops(shapeID) - 1  # -1 because first stop is the start of the journey

    vehicleType = fetchedDelays["route_type"]

    rawDelays = []

    kordisID = fetchedDelays["kordis_id"]
    lineID, benID = kordisID.split("/")

    # for delay in fetchedDelays:
    for date, values in fetchedDelays["data"].items():
        if values:
            rawDelays.append({
                date: values
            })

    if avgDelay:
        return shapeID, False, getAvgDelaysForPred(rawDelays, stopCount)

    avgDelays = getMedianDelays(rawDelays, stopCount)

    return shapeID, avgDelays, vehicleType, benID, lineID


def getAvgDelayAsPrediction(transport, depTime, predictionDay):
    return getDataAboutTransport(transport, depTime, predictionDay, True)


# Main function to get shapeID, avgDelay, vehicleType and realtime delays for given transport and departure time
def getShapeAndDelay(transport, depTime, predictionDay):
    returnShapeID, avgDelays, vehicleType, benID, lineID = getDataAboutTransport(transport, depTime, predictionDay, False)

    rawDelays = []

    if benID != 0:
        realtime, rawDelays = isRealTimePrediction(predictionDay, depTime, benID, lineID)

    parsedActualDelays = None

    if rawDelays is not None:
        if len(rawDelays) > 0:
            parsedActualDelays, realtime = mapDelays(rawDelays, returnShapeID)
    else:
        parsedActualDelays = None

    return returnShapeID, avgDelays, vehicleType, parsedActualDelays