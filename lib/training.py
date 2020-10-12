# -*- coding: utf-8 -*-

"""Script to generate an automatic report with dataset analysis"""

import pandas as pd
from utils import generate_report, preprocessing_data, Network
from torch import tensor, cuda, max, unique, true_divide, save
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    DEVICE = "cuda" if cuda.is_available() else "cpu"
    PATH_DATA = "../data/"
    NB_EPOCHS = 30
    LEARNING_RATE = 0.0001

    train_lab = pd.read_csv(PATH_DATA + 'train_labels.csv')["status_group"]
    train_val = pd.read_csv(PATH_DATA + 'train_values.csv')

    train_set = pd.concat([train_val, train_lab], axis=1)

    train_set = preprocessing_data(train_set)

    train_set, test_set = train_test_split(train_set, test_size=0.2)

    train_target = tensor(train_set['status_group'].values)
    train = tensor(train_set.drop('status_group', axis=1).values)
    train_tensor = data_utils.TensorDataset(train, train_target)
    train_loader = data_utils.DataLoader(dataset=train_tensor,
                                         batch_size=10,
                                         shuffle=True)

    # Validation
    test_target = tensor(test_set['status_group'].values)
    test = tensor(test_set.drop('status_group', axis=1).values)
    test_tensor = data_utils.TensorDataset(test, test_target)
    test_loader = data_utils.DataLoader(dataset=test_tensor,
                                        batch_size=10,
                                        shuffle=True)

    model = Network(len(train_set.columns)-1)
    model.to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')

    classes, rep = unique(train_target, sorted=True, return_counts=True)
    weights = true_divide(rep.sum(), rep).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    for epoch in range(NB_EPOCHS):
        loss_train, loss_test = 0, 0
        acc_test = 0
        model.train()
        for values, labels in train_loader:
            values, labels = values.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            pred = model(values.float())
            loss = criterion(pred, labels)
            loss.backward()
            opt.step()
            loss_train += loss.item()

        model.eval()
        for values, labels in test_loader:
            values, labels = values.to(DEVICE), labels.to(DEVICE)
            pred = model(values.float())
            loss = criterion(pred, labels)
            loss_test += loss.item()
            _, predicted = max(pred, dim=1)
            acc_test += (predicted == labels).sum().item()

        loss_train /= len(train_loader)
        loss_test /= len(test_loader)
        scheduler.step(loss_test)
        print(f'| Epoch: {epoch+1} | Train Loss: {loss_train:.3f} |'
              f"Test Loss: {loss_test:.3f} | Acc Test: {acc_test:.0f}"
              f"/{len(test_set):.0f}")

    CHECKPOINT = {'model': model,
                  'state_dict': model.state_dict(),
                  'optimiser': opt.state_dict()}
    save(CHECKPOINT['model'], '../data/model/model.pth')
    save(CHECKPOINT['state_dict'], '../data/model/state.pth')
