import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
from dataprocess import HousePriceDataset
from model import HousePriceModel

parser = argparse.ArgumentParser(description='House Price Prediction Regressor Model')
parser.add_argument('-epochs', default=300, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    train_dataset = HousePriceDataset('data/data_set_train.xlsx', label=True)
    test_dataset = HousePriceDataset('data/data_set_test.xlsx')
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    n_samples = len(train_dataset)
    n_features = train_dataset.get_feature_dim()
    model = HousePriceModel(n_features)
    criterion = nn.MSELoss() # Try different loss functions
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

    train_losses = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            x, lbl = data
            output = model(x)
            loss = torch.sqrt(criterion(torch.log(output), torch.log(lbl)))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        else: # Executes after the loop completes normally
            train_losses.append(train_loss/n_samples)
            avg_loss = train_loss/n_samples
            if avg_loss <= 0.001:
                break
            print(">> Epoch: {}/{} ".format(epoch+1, args.epochs), end='')
            print(" Training Loss: {:.3f} ".format(train_loss/n_samples))

    test_loss = 0
    accuracy = 0
    prices = []
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(test_loader):
            test_x, _ = data
            output = model(x)
            prices.append(output)
    print('[Price]\n', prices)
