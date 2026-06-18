# Delay Predictor for South Moravian Region

__Author:__ Martin Kováčik (xkovacm01)

## Instalation

To download all the project dependencies listed [here](./requirements.txt), follow these steps:

### Creating a virtual environment

```shell
python3 -m venv venv
```

### Then activate

macOS/Linux
```shell
source ./venv/bin/activate
```

Windows (CMD)

```shell
venv\Scripts\activate
```

Windows (PowerShell)

```shell
venv\Scripts\Activate.ps1
```

### Installing dependencies

```shell
pip3 install -r requirements.txt
```

## Download data

This script downloads and parses the data, then saves it to the __dataset__ folder in a folder named after the date of the download. The data will be saved as .json files, and finally, the normalized data for the neural network will be saved to the __date__ folder as __dataset.pt__.


|  Arguments   |  Type  |    Description   |
|--------------|--------|------------------|
| -d or --days |   int  |How many days from today to download |
| -r or --range |  string | Select date range for fetch (example: "[2026-01-29,2026-01-30]") |

If the script is run without any arguments, it will download yesterday's data.

```shell
python3 -m NN.fetchDelays  -r <date range> -d <days count>
```

### Example

With date range:

```shell
python3 -m NN.fetchDelays -r "[2026-4-21,2026-4-26]"
```

## Train neural network

If the dataset folder contains data for at least two days, you can run the training script using the following command.

```shell
python3 -m NN.trainNN
```

## Prediction

The predict.py script is used for prediction and accepts the following string:

__'{"visualization": boolean,"date": string date,"depTime": string time,"transport": {"line": line number,"route": route}}'__

### Example

```shell
python3 predict.py '{"visualization": true,"date": "2026-4-30","depTime": "12:21:00","transport": {"line": "1","route": "Řečkovice -> Rakovecká"}}'
```

Alternatively, you can run the predictor as an API using __uvicorn__

```shell
uvicorn api:app
```

You can then send a POST request to ```http://127.0.0.1:8000/predict``` with the route request in the following format:

```json
{
    "visualization": false,
    "date": "2026-4-30",
    "depTime": "12:21:00",
    "transport": {
        "line": "1",
        "route": "Řečkovice -> Rakovecká",
        "stop": "Tylova"
    }
}
```

#### Return object

Example of an object returned from a prediction for the entire route of a connection

```json
{
    "shape": 17477,
    "realtime": false,
    "prediction": {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 1,
        "7": 2,
        "8": 2,
        "9": 1,
        "10": 1
    }
}
```

Example of an object returned from the prediction for the selected stop

```json
{
    "prediction": 0
}
```

## API keys

This application requires API keys for Lissy, Ben, and OpenWeatherAPI. To obtain keys for Lissy and Ben, you must contact the [Dexter@FIT](https://dexter.fit.vutbr.cz). For OpenWeatherAPI, you can also use your own API key, which you can obtain after registering on [OpenWeatherAPI](https://openweathermap.org/api). The file containing the API keys is located in the [here](./apiFolder/apiKeys.py) folder.
