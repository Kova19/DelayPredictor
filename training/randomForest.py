import time
from datetime import datetime
from pathlib import Path

import json
import torch
import joblib

from logger.log import log

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


start = time.time()

logModel = "RandomForest"
version = "V2"

fileName = f"delayPredictorRandomForest{version}.joblib"


def getVocabSizes():
    with open("normalizers/LinesVocab.json", encoding="utf8") as f:
        linesVocab = json.load(f)

    return {name: len(mapping) for name, mapping in linesVocab.items()}


def parse_dataset_day(path: Path) -> datetime:
    year, month, day = map(int, path.parent.name.split("-"))
    return datetime(year, month, day)


def get_dataset_paths():
    dataset_root = Path(__file__).resolve().parent.parent / "dataset"
    dataset_files = list(dataset_root.glob("*/dataset.pt"))
    dataset_files.sort(key=parse_dataset_day)
    return [str(path) for path in dataset_files]


def trainRandomForest():
    files = get_dataset_paths()

    log("", False, logModel, version, "w")

    log(
        f"----- Training log from day: "
        f"{datetime.today().strftime('%d-%m-%Y %H:%M:%S')} -----",
        False,
        logModel,
        version
    )

    if not files:
        log("No dataset files found in dataset/*/dataset.pt", False, logModel, version)
        return

    if len(files) < 2:
        log(
            "Need at least 2 dataset days for temporal train/validation split",
            False,
            logModel,
            version
        )
        return

    log("Dataset trained on these days:", False, logModel, version)

    for path in files:
        log(Path(path).parent.name, True, logModel, version)

    log("\n", False, logModel, version)

    # Split the files into training and testing sets
    splitIndex = int(len(files) * 0.8)
    splitIndex = max(1, min(splitIndex, len(files) - 1))

    trainFiles = files[:splitIndex]
    testFiles = files[splitIndex:]

    log("Training days:", False, logModel, version)
    log(
        ", ".join([Path(path).parent.name for path in trainFiles]),
        False,
        logModel,
        version
    )

    log("Validation days (latest):", False, logModel, version)
    log(
        ", ".join([Path(path).parent.name for path in testFiles]),
        False,
        logModel,
        version
    )

    # Load datasets
    trainRawDataset = [torch.load(f) for f in trainFiles]
    testRawDataset = [torch.load(f) for f in testFiles]

    total_records = sum(
        len(ds) for ds in trainRawDataset + testRawDataset
    )

    log(f"Total sections count: {total_records}", False, logModel, version)

    trainTensor = torch.cat(trainRawDataset, dim=0).float()
    testTensor = torch.cat(testRawDataset, dim=0).float()

    del trainRawDataset
    del testRawDataset

    if trainTensor.size(0) == 0 or testTensor.size(0) == 0:
        log(
            "Temporal split produced empty train/validation tensor",
            False,
            logModel,
            version
        )
        return

    # X = features, y = target
    X_train = trainTensor[:, :-1].numpy()
    y_train = trainTensor[:, -1].numpy()

    X_test = testTensor[:, :-1].numpy()
    y_test = testTensor[:, -1].numpy()

    log(f"X_train shape: {X_train.shape}", False, logModel, version)
    log(f"y_train shape: {y_train.shape}", False, logModel, version)
    log(f"X_test shape: {X_test.shape}", False, logModel, version)
    log(f"y_test shape: {y_test.shape}", False, logModel, version)

    # Random Forest
    log("Creating Random Forest model...", False, logModel, version)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=30,
        min_samples_split=60,
        max_features=0.5,
        max_samples=0.25,
        n_jobs=8,
        random_state=42,
        bootstrap=True,
    )

    log("Fitting model...", False, logModel, version)

    model.fit(X_train, y_train)

    log("Model fitted.", False, logModel, version)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training metrics
    trainMSE = mean_squared_error(y_train, y_train_pred)
    trainMAE = mean_absolute_error(y_train, y_train_pred)
    trainR2 = r2_score(y_train, y_train_pred)

    # Validation metrics
    testMSE = mean_squared_error(y_test, y_test_pred)
    testMAE = mean_absolute_error(y_test, y_test_pred)
    testR2 = r2_score(y_test, y_test_pred)

    # Logging
    log("", False, logModel, version)

    log("---- Training results ----", False, logModel, version)

    log(f"Training MSE: {trainMSE:.3f}", False, logModel, version)
    log(f"Training MAE: {trainMAE:.3f}", False, logModel, version)
    log(f"Training R²:  {trainR2:.3f}", False, logModel, version)

    log("", False, logModel, version)

    log("---- Validation results ----", False, logModel, version)

    log(f"Testing MSE: {testMSE:.3f}", False, logModel, version)
    log(f"Testing MAE: {testMAE:.3f}", False, logModel, version)
    log(f"Testing R²:  {testR2:.3f}", False, logModel, version)

    # Feature importance
    log("", False, logModel, version)
    log("---- Feature importance ----", False, logModel, version)

    for index, importance in enumerate(model.feature_importances_):
        log(
            f"Feature {index}: {importance:.6f}",
            False,
            logModel,
            version,
        )

    # Save model
    try:
        joblib.dump(model, f"./models/{fileName}")

        log(f"Model saved as {fileName}", False, logModel, version)

    except Exception as e:
        log(f"Error while saving model: {e}", False, logModel, version)

    log("---- End training ----", False, logModel, version)


def main():
    trainRandomForest()

    end = time.time()

    log(
        f"Training was running: {((end - start) / 60):.2f} minutes",
        False,
        logModel,
        version
    )


if __name__ == "__main__":
    main()
