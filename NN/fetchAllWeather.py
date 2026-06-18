'''
Bachelor thesis FIT VUT
Author: Martin Kováčik (xkovacm01)
Date: 1.3.2026

Fetch script for fetching weather data from the API
'''

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, TypedDict
from collections import OrderedDict

import requests

from apiFolder.apiKeys import BENWEATHER, KEY, VALUE

api_key = KEY
api_value = VALUE
ben_weather = BENWEATHER

weatherAPIURL = "https://dexter.fit.vutbr.cz/lissy/api/weather/data"
weatherNewApiUrl = "https://walter.fit.vutbr.cz/ben/records/openWeather"


class weatherParsed(TypedDict):
    temp: float
    visibility: int
    windSpeed: float
    humidity: int
    rain1H: float
    snow1H: float
    group: str


# Convert timestamp to time string
def convertTimeStamp(timeToCnvt: int) -> str:
    timeMs = int(timeToCnvt) / 1000
    finalTime = datetime.fromtimestamp(timeMs)
    return finalTime.strftime("%H:%M:%S")


# Decode weather group from API to string
def decodeGroup(groupId: int) -> str:
    if groupId == 800:
        return "Clear"
    elif 801 <= groupId <= 804:
        return "Clouds"
    elif 200 <= groupId <= 232:
        return "Thunderstorm"
    elif 300 <= groupId <= 321:
        return "Drizzle"
    elif 500 <= groupId <= 531:
        return "Rain"
    elif 600 <= groupId <= 622:
        return "Snow"
    elif 700 <= groupId <= 781:
        return "Dist"
    else:
        return "Unknown"


# Parse weather data for fetching
def parseForFetching(data: Any) -> list[Any]:
    returnData: list[Any] = []

    for ts_epoch_ms, stations_dict in data.items():
        time_str = convertTimeStamp(ts_epoch_ms)

        time_obj = {}

        for station_id, station_data in stations_dict.items():
            if station_data is not None:
                time_obj[str(station_id)] = {
                    "temp": station_data["main"]["temp"],
                    "visibility": station_data.get("visibility", 10000),
                    "windSpeed": station_data["wind"]["speed"],
                    "humidity": station_data["main"]["humidity"],
                    "snow1H": station_data.get("snow", {}).get("1h", 0.0),
                    "rain1H": station_data.get("rain", {}).get("1h", 0.0),
                    "group": decodeGroup(station_data["weather"][0]["id"]),
                }
            else:
                time_obj[str(station_id)] = None

        returnData.append({time_str: time_obj})

    return returnData


# Parse weather data for fetching with new endpoint
def parseForFetchingNewURL(data):
    try:
        result = {}
        stations = {}

        for item in data:
            dt = datetime.fromisoformat(item["ben"]["timestamp"])
            timeKey = dt.strftime("%H:%M:%S")

            stationKey = item.get("ben", {}).get("key", "0")

            if timeKey not in result:
                result[timeKey] = {}

            result[timeKey][stationKey] = {
                "temp": item["main"]["temp"],
                "visibility": item.get("visibility", 10000),
                "windSpeed": item["wind"]["speed"],
                "humidity": item["main"]["humidity"],
                "snow1H": item.get("snow", {}).get("1h", 0.0),
                "rain1H": item.get("rain", {}).get("1h", 0.0),
                "group": item["weather"][0]["main"] if item.get("weather") else None
            }
            lat = item.get("coord", {}).get("lat")
            lon = item.get("coord", {}).get("lon")
            stations[stationKey] = [lat, lon]

        sortedResult = OrderedDict(sorted(result.items(), key=lambda x: x[0]))
        return [sortedResult], stations
    except Exception as e:
        print(f"Error parsing weather data: {e}")


#  Return list[weatherParsed] if single is false, else weatherParsed
def parse(data: Any, single: bool):
    returnData: list[Any] = []

    for n in data:
        for j in data[n].values():
            obj = weatherParsed(
                temp=j["main"]["temp"],
                visibility=j.get("visibility", 10000),
                windSpeed=j["wind"]["speed"],
                humidity=j["main"]["humidity"],
                snow1H=j.get("snow", {}).get("1h", 0.0),
                rain1H=j.get("rain", {}).get("1h", 0.0),
                group=decodeGroup(j["weather"][0]["id"]),
            )
            returnData.append(obj)
            if single:
                return returnData[0]

    return returnData


def fetchWeather(fromTime: int, toTime: int, where: int) -> weatherParsed:
    headers = {api_key: api_value}
    params = {"from": fromTime, "to": toTime, "positionId": where}

    x = requests.get(weatherAPIURL, headers=headers, params=params)
    weatherData = parse(x.json(), True, False)

    return weatherData


def fetchWeatherByDay(dayStart: int):
    dayEnd = dayStart + 86399000  # + 24 hours (practily get one day)

    headers = {api_key: api_value}

    params = {
        "from": dayStart,
        "to": dayEnd,
    }

    x = requests.get(weatherAPIURL, headers=headers, params=params)
    weatherData = parseForFetching(x.json())

    return weatherData


def fetchWeatherByDayNewEndpoint(dayStart):
    try:
        dayFrom = dayStart.strftime("%Y-%m-%dT00:00:00")
        dayTo = dayStart.strftime("%Y-%m-%dT23:59:59")

        headers = {api_key: ben_weather}
        params = {"dateFrom": dayFrom, "dateTo": dayTo}

        x = requests.get(weatherNewApiUrl, params=params, headers=headers)
        obj = x.json()

        weatherData, stations = parseForFetchingNewURL(obj)

        return weatherData, stations
    except Exception as e:
        print(f"Error while fetching weathet for day {dayStart}: {e}")


def fetchAll():
    weatherData: Any
    dt = datetime.now(timezone.utc) - timedelta(days=1)
    epoch_ms = int(dt.timestamp() * 1000)

    headers = {api_key: api_value}
    params = {"from": 0, "to": epoch_ms, "positionId": 0}

    x = requests.get(weatherAPIURL, headers=headers, params=params)
    weatherData = parse(x.json(), False)
    with open("./weatherHistory.json", "w", encoding="utf-8") as f:
        print(json.dumps(weatherData, indent=2, ensure_ascii=False), file=f)


def main():
    fetchAll()


if __name__ == "__main__":
    main()
