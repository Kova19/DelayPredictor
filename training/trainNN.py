'''
Bachelor thesis FIT VUT
Author: Martin Kováčik (xkovacm01)
Date: 20.3.2026

Training script for the delay predictor neural network.
'''

import time
from datetime import datetime
from pathlib import Path
from logger.log import log

import torch
import json
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from traning.neuralNetwork import DelayPredictor

start = time.time()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logModel = "NeuralNetwork"
version = "V15.1"
fileName = f"delayPredictorModel{version}.pt"


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


def trainNN():
    files = get_dataset_paths()

    log("", False, logModel, version, "w")

    log(
        f"----- Training log from day: {
            datetime.today().strftime('%d-%m-%Y %H:%M:%S')} -----",
        False,
        logModel,
        version
    )

    log("dataset trained on this days:", False, logModel, version)

    if not files:
        log("No dataset files found in dataset/*/dataset.pt", False, logModel, version)
        return

    if len(files) < 2:
        log("Need at least 2 dataset days for temporal train/validation split", False, logModel, version)
        return

    log("---- Start training ----", False, logModel, version)

    model = DelayPredictor(vocab_sizes=getVocabSizes()).to(device)

    epochs = 1000

    # Criterion for loss calculation and MAE for earlyStop
    criterion = nn.MSELoss()
    mae = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=2e-2)
    # scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-8
    )

    splitIndex = int(len(files) * 0.8)
    splitIndex = max(1, min(splitIndex, len(files) - 1))

    trainFiles = files[:splitIndex]
    testFiles = files[splitIndex:]

    log("Training days:", False, logModel, version)
    log(", ".join([Path(path).parent.name for path in trainFiles]), False, logModel, version)
    log("Validation days (latest):", False, logModel, version)
    log(", ".join([Path(path).parent.name for path in testFiles]), False, logModel, version)

    # Load and concat train/test datasets separately to preserve temporal order by day.
    trainRawDataset = [torch.load(f) for f in trainFiles]
    testRawDataset = [torch.load(f) for f in testFiles]

    total_records = sum(len(ds) for ds in trainRawDataset + testRawDataset)
    log(f"Total sections count: {total_records}", False, logModel, version);

    trainTensor = torch.cat(trainRawDataset, dim=0).float()
    testTensor = torch.cat(testRawDataset, dim=0).float()
    del trainRawDataset
    del testRawDataset

    if trainTensor.size(0) == 0 or testTensor.size(0) == 0:
        log("Temporal split produced empty train/validation tensor after filtering", False, logModel, version)
        return

    # X is the input data and y the target value
    X_train = trainTensor[:, :-1].to(device)
    y_train = trainTensor[:, -1].to(device).view(-1, 1)
    X_test = testTensor[:, :-1].to(device)
    y_test = testTensor[:, -1].to(device).view(-1, 1)

    # Batch dataset
    batchSize = 256
    trainDataset = TensorDataset(X_train, y_train)
    testDataset = TensorDataset(X_test, y_test)
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

    bestMae = float("inf")
    notImprovedEpochs = 0
    patience = 7  # 6 becuase lr is getting lower each 5 epochs

    for i in range(epochs):
        trainLossSum = 0.0
        trainMaeSum = 0.0
        trainSamples = 0
        testLossSum = 0.0
        testMaeSum = 0.0
        testSamples = 0

        # Train the model
        for XBatch, yBatch in trainLoader:
            model.train()
            XBatch = XBatch.to(device)
            yBatch = yBatch.to(device)
            batchSizeNow = yBatch.size(0)

            optimizer.zero_grad()
            yPrediction = model(XBatch)

            loss = criterion(yPrediction, yBatch)
            maeLoss = mae(yPrediction, yBatch)

            trainLossSum += loss.item() * batchSizeNow
            trainMaeSum += maeLoss.item() * batchSizeNow
            trainSamples += batchSizeNow

            loss.backward()
            optimizer.step()

        # Test the model
        for XBatch, yBatch in testLoader:
            model.eval()
            XBatch = XBatch.to(device)
            yBatch = yBatch.to(device)
            batchSizeNow = yBatch.size(0)

            yPrediction = model(XBatch)

            test_loss = criterion(yPrediction, yBatch)
            testMae = mae(yPrediction, yBatch)

            testLossSum += test_loss.item() * batchSizeNow
            testMaeSum += testMae.item() * batchSizeNow
            testSamples += batchSizeNow

        trainLoss = trainLossSum / max(trainSamples, 1)
        trainMAE = trainMaeSum / max(trainSamples, 1)
        testLoss = testLossSum / max(testSamples, 1)
        testMAE = testMaeSum / max(testSamples, 1)

        # update learning rate
        scheduler.step(testMAE)

        log(f"Learning rate is set to {optimizer.param_groups[0]['lr']}", False, logModel, version)
        log(f"Epoch: {i}, training MSE loss: {trainLoss:.3f}", False, logModel, version)
        log(f"Epoch: {i}, training MAE loss: {trainMAE:.3f}", False, logModel, version)
        log(f"Epoch: {i}, testing MSE loss: {testLoss:.3f}", False, logModel, version)
        log(f"Epoch: {i}, testing MAE loss: {testMAE:.3f}", False, logModel, version)
        log("", False, logModel, version)

        # Primitive earlyStopping
        if testMAE < bestMae:
            bestMae = testMAE
            bestModel = model
            notImprovedEpochs = 0
        else:
            notImprovedEpochs += 1

        if notImprovedEpochs >= patience:
            log(
                f"Early stopping at epoch {
                    i} because of no improvement in the last {patience} epochs",
                False,
                logModel,
                version
            )
            break

    log("---- End training ----", False, logModel, version)

    try:
        # Save portable weights only (no class/module pickling).
        stateDictCpu = {k: v.detach().cpu() for k, v in bestModel.state_dict().items()}
        torch.save(stateDictCpu, f"./models/{fileName}")
        log(f"Model saved as {fileName}", False, logModel, version)
    except Exception as e:
        log(f"Error while saving model: {e}", False, logModel, version)


def main():
    trainNN()
    end = time.time()
    log(f"Training was running: {((end - start)/60):.2f} minutes", False, logModel, version)


if __name__ == "__main__":
    main()
