"""
Bachelor thesis FIT VUT
Author: Martin Kováčik (xkovacm01)
Date: 10.3.2026

Fetch script for fetching all lines, routes and stops from the API
"""

urlRoutes = "https://dexter.fit.vutbr.cz/lissy/api/delayTrips/getAvailableRoutes"
urlTrips = "https://dexter.fit.vutbr.cz/lissy/api/delayTrips/getAvailableTrips"
urlForStops = "https://dexter.fit.vutbr.cz/lissy/api/shapes/getShape"
urlForDelays = "https://dexter.fit.vutbr.cz/lissy/api/delayTrips/getTripData"
urlForShapes = "https://dexter.fit.vutbr.cz/lissy/api/shapes/getShapes"
urlForShape = "https://dexter.fit.vutbr.cz/lissy/api/shapes/getShape"
urlForRealtimeDelays = "https://walter.fit.vutbr.cz/ben/records/vehiclePositions"

urlBenWeather = "https://walter.fit.vutbr.cz/ben/records/openWeather"
urlForWeather = "https://api.openweathermap.org/data/2.5/forecast"
