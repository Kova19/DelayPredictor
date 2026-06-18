'''
Bachelor thesis FIT VUT
Author: Martin Kováčik (xkovacm01)
Date: 20.4.2026

Script for fetch shape ID and avgDelays for given transport and departure time.
'''

from datetime import datetime, timedelta, time

import requests
import numpy as np
from zoneinfo import ZoneInfo

from apiFolder.apiKeys import KEY, VALUE, BENWEATHER
from NN.fetchDelays import fixDelays
from geopy.distance import geodesic

urlForShapes = "https://dexter.fit.vutbr.cz/lissy/api/delayTrips/getShape"
urlForRoutes = "https://dexter.fit.vutbr.cz/lissy/api/delayTrips/getAvailableRoutes"
urlForTrips = "https://dexter.fit.vutbr.cz/lissy/api/delayTrips/getAvailableTrips"
urlForDelays = "https://dexter.fit.vutbr.cz/lissy/api/delayTrips/getTripData"
urlForRealtimeDelays = "https://walter.fit.vutbr.cz/ben/records/vehiclePositions"

headers = {KEY: VALUE}
headersBen = {KEY: BENWEATHER}

# Find short name of line
def findLineID(lines, line):
    for lookLine in lines:
        if lookLine["route_short_name"] == line:
            return lookLine["id"], lookLine["route_type"]
    return -1, -1

# Try to find line first try today - 7 else try for 20 days back
def findLine(line, predictionDay):
    date = datetime.strptime(predictionDay, "%Y-%m-%d")
    lookupDate = date - timedelta(days=7)
    lookDay = lookupDate.day
    lookMonth = lookupDate.month - 1
    lookYear = lookupDate.year

    params = {
        "dates": f'[["{lookYear}-{lookMonth}-{lookDay}","{lookYear}-{lookMonth}-{lookDay}"]]'
    }
    try:
        x = requests.get(urlForRoutes, params=params, headers=headers)
        obj = x.json()

        lineID, vehicleType = findLineID(obj, line)
        if lineID is -1:
            for i in range(20):
                lookupDate = datetime.now() - timedelta(days=i)
                lookDay = lookupDate.day
                lookMonth = lookupDate.month - 1
                lookYear = lookupDate.year
                params = {
                    "dates": f'[["{lookYear}-{lookMonth}-{lookDay}","{lookYear}-{lookMonth}-{lookDay}"]]'
                }

                x = requests.get(urlForRoutes, params=params, headers=headers)
                obj = x.json()
                lineID, vehicleType = findLineID(obj, line)

                if lineID is not -1:
                    break
                else:
                    continue
        if lineID == -1 or vehicleType == -1:
            return -1, -1

        return lineID, vehicleType

    except Exception as e:
        print(e)


# Try to find tripID, shapeID and externalTrioID (benID)
def findTrip(routes, depTime, givenRoute):
    if routes is []:
        return -1, -1, -1

    for route in routes:
        for trip in route["trips"]:
            if trip["dep_time"] == depTime and route["stops"] == givenRoute:
                return trip["id"], route["shape_id"], trip["externalTripId"]
    return -1, -1, -1


# Try to find trip, first try for today - 7 else try for 20 days back
def findTripID(lineID, depTime, predictionDay, route):
    tripIDField = []
    shapeIDField = []
    benIDField = []
    date = datetime.strptime(predictionDay, "%Y-%m-%d")
    lookupDate = date - timedelta(days=7)
    lookDay = lookupDate.day
    lookMonth = lookupDate.month - 1
    lookYear = lookupDate.year

    params = {
        "dates": f'[["{lookYear}-{lookMonth}-{lookDay}","{lookYear}-{lookMonth}-{lookDay}"]]',
        "route_id": lineID,
    }

    try:
        x = requests.get(urlForTrips, params=params, headers=headers)
        obj = x.json()

        tripID, shapeId, benID = findTrip(obj, depTime, route)

        if tripID != -1:
            tripIDField.append(tripID)
        if shapeId != -1:
            shapeIDField.append(shapeId)
        if benID != -1:
            benIDField.append(benID)

        # if tripID is -1:
        for i in range(20):
            lookupDate = datetime.now() - timedelta(days=i)
            lookDay = lookupDate.day
            lookMonth = lookupDate.month - 1
            lookYear = lookupDate.year
            params = {
                "dates": f'[["{lookYear}-{lookMonth}-{lookDay}","{lookYear}-{lookMonth}-{lookDay}"]]',
                "route_id": lineID,
            }

            x = requests.get(urlForTrips, params=params, headers=headers)
            obj = x.json()
            tripID, shapeId, benID = findTrip(obj, depTime, route)

            if tripID != -1:
                tripIDField.append(tripID)
            if shapeId != -1:
                shapeIDField.append(shapeId)
            if benID != -1:
                benIDField.append(benID)

        return tripIDField, shapeIDField, benIDField

    except Exception as e:
        print(e)


# Count stops for given shapeID
def countStops(shapeID):
    try:
        params = {
            "shape_id": shapeID
        }

        x = requests.get(urlForShapes, headers=headers, params=params)
        obj = x.json()

        return len(obj["stops"])

    except Exception as e:
        print(e)
        return -1


# Get average delays for given tripID
def getAvgDelays(data, sectionCnt):
    try:
        #  return {} if data are not provided
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
                        if sectionData:
                            values.append(list(sectionData.values())[-1])

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


# Get delays for last 21 days for given tripID
def getDelays(tripID, sectionCount):
    dateTo = datetime.now()
    # Get JS date
    dayTo = dateTo.day
    monthTo = dateTo.month - 1
    yearTo = dateTo.year

    dateFrom = datetime.now() - timedelta(days=21)
    dayFrom = dateFrom.day
    monthFrom = dateFrom.month - 1
    yearFrom = dateFrom.year

    fetchDelays = []

    try:
        for trip in tripID:
            params = {
                "dates": f'[["{yearFrom}-{monthFrom}-{dayFrom}", "{yearTo}-{monthTo}-{dayTo}"]]',
                "trip_id": trip
            }

            x = requests.get(urlForDelays, headers=headers, params=params)
            obj = x.json()
            if obj is not None and obj != {}:
                fetchDelays.append(obj)

        return getAvgDelays(fetchDelays, sectionCount)

    except Exception as e:
        print(e)
        return {}


# Get route key for given line, try first for today - 7 else try for 20 days back
def getRouteKey(line):
    lookDay = datetime.today() - timedelta(days=7)
    params = {
        "dates": f'[["{lookDay.year}-{lookDay.month - 1}-{lookDay.day}","{lookDay.year}-{lookDay.month - 1}-{lookDay.day}"]]'
    }

    try:
        x = requests.get(urlForRoutes, headers=headers, params=params)
        obj = x.json()

        if obj == {}:
            for i in range(1, 5):
                lookDay = datetime.today() - timedelta(days=7*i)
                params = {
                    "dates": f'[["{lookDay.year}-{lookDay.month - 1}-{lookDay.day}","{lookDay.year}-{lookDay.month - 1}-{lookDay.day}"]]'
                }
                x = requests.get(urlForRoutes, headers=headers, params=params)
                obj = x.json()
                if obj != {}:
                    break
        for route in obj:
            if route["route_short_name"] == line:
                notParserKey = route["route_id"]
                result = notParserKey[notParserKey.find('L')+1: notParserKey.find('D')]
                return result

        return None
    except Exception as e:
        print(e)
        return None


# Convert departure time to UTC format for given time in local timezone
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
            if data["ben"]["key"] == key and int(data["RouteID"]) == benRouteID:
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
def isRealTimePrediction(predictionDay, depTime, line, benID):
    if benID == {}:
        return False, None
    
    depTimeUTC = convertUTCTime(depTime)

    depTimePrediction = datetime.strptime(depTimeUTC, "%Y-%m-%dT%H:%M:%S.000Z")

    predDate = datetime.strptime(predictionDay, "%Y-%m-%d")
    predDate = predDate.replace(hour=depTimePrediction.hour, minute=depTimePrediction.minute, second=depTimePrediction.second)


    today = datetime.utcnow()
    thresHoldTime = predDate + timedelta(hours=2)

    predDate = datetime.fromisoformat(depTimeUTC.replace("Z", "+00:00")).replace(tzinfo=None)


    if  predDate < today < thresHoldTime:
        key = getRouteKey(line)
        benRouteID = benID
        actualVehiclePositions = getRealtimeDelays(benRouteID, key, predDate)

        return True, actualVehiclePositions
    else:
        return False, None


# Map realtime delays to stops
def mapDelays(rawDelays, shapeID):
    params = {
        "shape_id": shapeID
    }

    x = requests.get(urlForShapes, headers=headers, params=params)
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


# Main function to get shapeID, avgDelay, vehicleType and realtime delays for given transport and departure time
def getShapeAndDelay(transport, depTime, predictionDay):
    line = transport["line"]
    route = transport["route"]

    lineID, vehicleType = findLine(line, predictionDay)
    if lineID is -1:
        return -1, -1, -1, -1

    tripID, shapeId, benID = findTripID(lineID, depTime, predictionDay, route)

    if tripID == [] or shapeId == [] or tripID == -1:
        return -2, -2, -2, -2

    rawDelays = None

    if benID:
        for id in benID:
            if id is not None:
                realtime, rawDelays = isRealTimePrediction(predictionDay, depTime, line, id)
            if rawDelays is not None:
                break

    parsedActualDelays = None

    if rawDelays is not None:
        for shapes in shapeId:
            if shapes is not None:
                if len(rawDelays) > 0:
                    parsedActualDelays, realtime = mapDelays(rawDelays, shapes)
            if parsedActualDelays is not None:
                break
    else:
        parsedActualDelays = None

    for shape in shapeId:
        if shape is not None:
            sectionCount = countStops(shape) - 1
            returnShapeID = shape
        if sectionCount > 0:
            break

    avgDelay = getDelays(tripID, sectionCount)
    if avgDelay == {}:
        avgDelay = -10
    if avgDelay is None:
        avgDelay = -10

    return returnShapeID, avgDelay, vehicleType, parsedActualDelays
