'''
Bachelor thesis FIT VUT
Author: Martin Kováčik (xkovacm01)
Date: 2.3.2026

Fetch script for fetching delay data from the API and saving it in the dataset folder.
'''

import argparse
import concurrent.futures
import json
import os
from datetime import datetime, timedelta

import holidays
import numpy as np
import requests
import torch
from geopy.distance import geodesic

from apiFolder.apiKeys import KEY, VALUE
from encoding import newEncode
from fetchers.fetchAllWeather import fetchWeatherByDayNewEndpoint
from fetchers.fetchStopsCords import getAllStops
from constants.constants import urlRoutes, urlTrips, urlForStops, urlForDelays

with open("normalizers/weatherNormalizer.json", encoding="utf8") as f:
    weatherVocab = json.load(f)

with open("normalizers/LinesVocab.json", encoding="utf8") as f:
    linesVocab = json.load(f)

czechHolidays = holidays.CZ()

headers = {KEY: VALUE}


# First peak is from 7:00 to 8:30
def firstPeak(mainTime: str):
    start = datetime.strptime("7:00", "%H:%M").time()
    end = datetime.strptime("8:30", "%H:%M").time()
    comTime = datetime.strptime(mainTime, "%H:%M:%S").time()

    if start <= comTime <= end:
        return True
    else:
        return False


# Second peak is from 15:30 to 17:30
def secondPeak(mainTime: str):
    start = datetime.strptime("15:30", "%H:%M").time()
    end = datetime.strptime("17:30", "%H:%M").time()
    comTime = datetime.strptime(mainTime, "%H:%M:%S").time()

    if start <= comTime <= end:
        return True
    else:
        return False


# Time to seconds
def time_to_seconds(t: str) -> int:
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s


# Find the date of dataset from the path
def findWeatherForTime(target_time: str, station_id: str, weatherForDay):
    target_sec = time_to_seconds(target_time)

    closest_data = None
    min_diff = float("inf")

    # Find best result for given time and station
    for entry in weatherForDay:
        for timestamp, stations in entry.items():
            if station_id in stations:
                diff = abs(time_to_seconds(timestamp) - target_sec)

                if diff < min_diff:
                    min_diff = diff
                    closest_data = stations[station_id]

    if closest_data is not None:
        return {station_id: closest_data}

    return None


# Check if the given date is a czech holiday
def isHoliday(date: str) -> bool:
    return date in czechHolidays


# Get stop cords
def getStopCords(name: str, allStops):
    #  Find and return the coords of the stop
    for stop in allStops:
        if stop["StopName"] == name:
            return stop["coords"]
    return None


# Get weather for given time and stop
def getWeatherForStop(
    stop: str, timeStr: str, weatherStations, weatherForDay, allStops
):
    try:
        stopCords = getStopCords(stop, allStops)
        bestId = 0  # 0 for fallback if there is no station or stop coords
        bestMetrs = float("inf")

        #  get the closest weather station to the stop
        for station_id, coords in weatherStations.items():
            tmp = geodesic(stopCords, coords).meters
            if tmp < bestMetrs:
                bestId = station_id
                bestMetrs = tmp

        #  Get weather for best station with closest time
        returnWeather = findWeatherForTime(timeStr, bestId, weatherForDay)
        returnData = next(iter(returnWeather.values()))

        return returnData

    except Exception as e:
        print(f"Error while fetching data for {timeStr}: {e}")


# Fix delay algorithm
def fixDelays(data):
    fixedDelays = {}
    nanCount = 0
    maxNan = 3
    outlierCount = 0
    maxOutlier = 3
    maxPlusDiff = 30
    maxMinusDiff = -15
    expectedFirstDelay = 0

    for index, delay in data.items():
        # First delay repair or filter
        if index == 0:
            if np.isnan(delay):
                nanCount += 1
                delay = expectedFirstDelay
            if delay > 100 or delay < -5:
                outlierCount += 1
                delay = expectedFirstDelay

        # Check for missing sections
        if np.isnan(delay):
            nanCount += 1
            delay = fixedDelays[index - 1]
        else:
            nanCount = 0

        if index != 0:
            # Check for outliers
            if delay > fixedDelays[index-1] + maxPlusDiff or delay < fixedDelays[index-1] + maxMinusDiff:
                delay = fixedDelays[index - 1]
                outlierCount += 1
            else:
                outlierCount = 0

        # Last check if the number of outliers and NaNs is too high
        # in this case the route is not fixable
        if outlierCount > maxOutlier or nanCount > maxNan:
            return {}

        fixedDelays[index] = delay

    return fixedDelays


# Get delays for given trip
def getDelays(id: int, sectionCnt: int, fetchDay):
    try:
        fetchDayDelay = {}

        todayFilter = f"{fetchDay.year}-{fetchDay.month - 1}-{fetchDay.day}"
        startDate = datetime.today() - timedelta(days=60)

        startDay = startDate.day
        startMonth = startDate.month - 1
        startYear = startDate.year

        dateRange = f'[["{startYear}-{startMonth}-{startDay}","{todayFilter}"]]'
        params = {"dates": dateRange, "trip_id": id}

        x = requests.get(urlForDelays, headers=headers, params=params)
        obj = x.json()

        if not obj:
            return {"delays": {}, "avgDelays": {}}

        if obj.get(todayFilter) is not None:
            todayData = obj[todayFilter]
        else:
            return {"delays": {}, "avgDelays": {}}

        # Fetch this day Delays and store it and fix errors
        i = 0
        while i < sectionCnt:
            if todayData.get(f"{i}") is not None:
                fetchDayDelay[i] = list(todayData[f"{i}"].values())[-1]

            #   If there is hole use nan value
            else:
                if i == 0:
                    fetchDayDelay[i] = 0
                else:
                    fetchDayDelay[i] = np.nan
            i += 1

        #  AvgDelays will be fixed
        avgDelays = getAvgDelays(obj, sectionCnt)

        #fixedTodayDelays = fixDelaysUsingKF(fetchDayDelay)
        fixedTodayDelays = fixDelays(fetchDayDelay)

        returnData = {"delays": fixedTodayDelays, "avgDelays": avgDelays}
        return returnData

    except Exception as e:
        print(f"Error while getting delays: {e}")
        return {"delays": {}, "avgDelays": {}}


# Get avgdelays
def getAvgDelays(data, sectionCnt: int):
    try:
        #  return {} if data are not provided
        if not data:
            return {}

        result = {}
        i = 0
        while i < sectionCnt:  # bcs index from 0
            values = []

            # Get all data from current section
            for day in data.values():
                #  Get all delays in this part of route
                if day.get(f"{i}") is not None:
                    values.append(list(day[f"{i}"].values())[-1])
                else:
                    continue

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
        return {}


# get stops
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


# Fetch trips and save it in dataset
def fetchTrips(route, dates, fetchDay, weatherStations, weatherForDay, allStops):
    objectForSave: list = []
    try:
        params = {"dates": dates, "route_id": route["id"]}

        x = requests.get(urlTrips, headers=headers, params=params)
        obj = x.json()

        print(f"Saving line: {route['route_short_name']}")
        for j in obj:
            stops = getStops(j["shape_id"])
            for trip in j["trips"]:
                fetchDelays = getDelays(trip["id"], int(len(stops)) - 1, fetchDay)

                if fetchDelays["avgDelays"] == {} or fetchDelays["delays"] == {}:
                    continue

                avgDelays = fetchDelays["avgDelays"]
                delays = fetchDelays["delays"]
                weather = getWeatherForStop(
                    stops[0], trip["dep_time"], weatherStations, weatherForDay, allStops
                )
                i = 0
                lastDelay = 0
                for n in stops[1:]:
                    fetchTranspData = {
                        "line": route["route_short_name"],
                        "route": j["stops"],
                        "vehicleType": route["route_type"],
                        "stop": n,
                    }
                    if delays.get(i) is not None:
                        fetchDelay = delays[i]
                        fetchAvgDelay = avgDelays[i]
                    else:
                        fetchDelay, avgDelays = lastDelay
                    i += 1
                    holiDay = isHoliday(
                        f"{fetchDay.year}-{fetchDay.month}-{fetchDay.day}"
                    )
                    dayOfWeek = fetchDay.strftime("%A")
                    depTime = trip["dep_time"]

                    tmp = {
                        "dayOfWeek": dayOfWeek,
                        "depTime": depTime,
                        "7:00-8:30": firstPeak(depTime),
                        "15:30-17:30": secondPeak(depTime),
                        "transport": fetchTranspData,
                        "stopIndex": i,
                        "stopsCount": len(stops) - 1,
                        "weather": weather,
                        "holiday": holiDay,
                        "prevDelay": lastDelay,
                        "avgDelay": fetchAvgDelay,
                        "delay": fetchDelay,
                    }
                    lastDelay = fetchDelay
                    objectForSave.append(tmp)

            print(f"Saved route: {j['shape_id']} (shapeID)")
        with open(f"./dataset/{fetchDay.year}-{fetchDay.month}-{fetchDay.day}/line{route['route_short_name']}.json",
            "w",
            encoding="utf-8",
        ) as f:
            print(json.dumps(objectForSave, indent=2, ensure_ascii=False), file=f)
        return objectForSave

    except Exception as e:
        print(f"Error while fetching trips id: {e}")


# Main function for fetching data and saving it in dataset
def fetch(fetchDay):
    dataset = []
    print(f"Fetching stops cords for day: {
          fetchDay.year}-{fetchDay.month}-{fetchDay.day}")
    allStops = getAllStops(f"{fetchDay.year}-{fetchDay.month - 1}-{fetchDay.day}")
    print(f"All stops cords fetched for day: {
          fetchDay.year}-{fetchDay.month}-{fetchDay.day}")

    weatherForDay, weatherStations = fetchWeatherByDayNewEndpoint(fetchDay)

    try:
        year = fetchDay.year
        month = fetchDay.month - 1
        day = fetchDay.day

        dates = f'[["{year}-{month}-{day}","{year}-{month}-{day}"]]'

        params = {
            "dates": dates,
        }
        x = requests.get(urlRoutes, headers=headers, params=params)
        routes = x.json()

        for route in routes:
            fetchData = fetchTrips(
                route, dates, fetchDay, weatherStations, weatherForDay, allStops
            )

            if fetchData == {} or fetchData is None:
                continue

            tensors = [
                newEncode(
                    data,
                    False,
                    linesVocab,
                    weatherVocab
                )
                for data in fetchData
            ]
            tensors = [t for t in tensors if t is not None]

            if tensors:
                dataset.append(torch.stack(tensors))

        dataset = torch.cat(dataset)
        torch.save(
            dataset,
            f"./dataset/{fetchDay.year}-{fetchDay.month}-{fetchDay.day}/dataset.pt",
        )

    except Exception as e:
        print(f"Error while fetching data: {e}")


# Argument parser
def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--days", help="How many days from today to download", type=int
    )

    parser.add_argument(
        "-r",
        "--range",
        help='Select date range for fetch (example: "[2026-01-29,2026-01-30]")',
        type=str,
    )
    return parser.parse_args()


# Get multiple days from range argument
def getMultipleDays(dayArgument):
    raw = dayArgument.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]

    start_str, end_str = raw.split(",")

    start = datetime.strptime(start_str.strip(), "%Y-%m-%d")
    end = datetime.strptime(end_str.strip(), "%Y-%m-%d")

    daysForFetch = []

    if start > end:
        raise ValueError("End day must be older")

    daysForFetch = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    return daysForFetch


# Fetch days with thread pool for faster fetching
def fetchDaysWithThreadPool(daysForFetch, maxWorkers=8):
    if not daysForFetch:
        return

    workerCount = min(maxWorkers, len(daysForFetch))
    print(f"Starting fetch with {workerCount} thread(s)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workerCount) as executor:
        futureToDay = {
            executor.submit(fetch, fetchDay): fetchDay for fetchDay in daysForFetch
        }

        for future in concurrent.futures.as_completed(futureToDay):
            fetchDay = futureToDay[future]
            try:
                future.result()
                print(f"Finished day: {fetchDay.year}-{fetchDay.month}-{fetchDay.day}")
            except Exception as e:
                print(f"Failed day {fetchDay.year}-{fetchDay.month}-{fetchDay.day}: {e}")


# Main function for fetching data and saving it in dataset
def main():
    daysForFetch = []
    args = parseArguments()

    if not args.range and not args.days:
        date = datetime.now() - timedelta(days=1)
        daysForFetch.append(date)

    if args.days is not None:
        print(args.days)
        for i in range(args.days):
            daysForFetch.append(datetime.now() - timedelta(days=i + 1))

    if args.range is not None:
        daysForFetch = getMultipleDays(args.range)

    for fetchDay in daysForFetch:
        fetchingDay = f"{fetchDay.year}-{fetchDay.month}-{fetchDay.day}"
        path = os.path.join("dataset", fetchingDay)
        os.makedirs(path, exist_ok=True)

    fetchDaysWithThreadPool(daysForFetch, maxWorkers=8)


if __name__ == "__main__":
    main()
